
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer



# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def _precompute_rope_cache(seq_len: int, head_dim: int, device: torch.device,
                           theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device)
    angles = torch.outer(positions, freqs)
    return angles.cos(), angles.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class RoPEAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width ({width}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self._rope_cos = None
        self._rope_sin = None

    def _ensure_rope_cache(self, seq_len: int, device: torch.device):
        if self._rope_cos is None or self._rope_cos.shape[0] < seq_len:
            cos, sin = _precompute_rope_cache(seq_len, self.head_dim, device)
            self._rope_cos = cos
            self._rope_sin = sin
        return self._rope_cos[:seq_len], self._rope_sin[:seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: [B, H, T, D]

        cos, sin = self._ensure_rope_cache(length, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.proj(attended)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        hidden = int(8 / 3 * width)
        hidden = (hidden + 7) // 8 * 8  # align to multiple of 8
        self.w1 = nn.Linear(width, hidden, bias=False)  # gate
        self.w2 = nn.Linear(hidden, width, bias=False)  # down-projection
        self.w3 = nn.Linear(width, hidden, bias=False)  # value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class ImprovedBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = RoPEAttention(width, heads)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
def _build_bigram_logp(vocab: int, beta: float = 50.0, alpha: float = 0.1):
    root = Path(__file__).resolve().parent
    tokenizer = Tokenizer.from_file(str(root / 'data' / 'tokenizer.json'))
    raw = (root / 'data' / 'wikitext_train.txt').read_text(encoding='utf-8')
    ids = torch.tensor(tokenizer.encode(raw).ids, dtype=torch.long)
    uni = torch.bincount(ids[1:], minlength=vocab).float()
    uni = (uni + alpha) / (uni.sum() + alpha * vocab)
    pair_ids = ids[:-1] * vocab + ids[1:]
    counts = torch.bincount(pair_ids, minlength=vocab * vocab).float().view(vocab, vocab)
    denom = counts.sum(dim=1, keepdim=True) + beta
    probs = (counts + beta * uni[None, :]) / denom
    return probs.log()


def _apply_local_bigram_cache(base_logp, ids, cache_lambda=0.08, cache_k=2.0):
    import math as _math
    B, T, V = base_logp.shape
    out = base_logp.clone()
    eps = 1e-10

    for b in range(B):
        ids_b = ids[b]  # [T]

        # Transitions: prev[i] -> next[i] for i in 0..T-2
        prev = ids_b[:-1]  # [T-1]
        nxt = ids_b[1:]    # [T-1]

        # mask[t, i] = 1 if transition i is available at position t AND prev[i] == ids[t]
        # Available: i < t (since ids[i+1] must be within ids[:t+1])
        t_idx = torch.arange(T, device=ids.device).unsqueeze(1)   # [T, 1]
        i_idx = torch.arange(T - 1, device=ids.device).unsqueeze(0)  # [1, T-1]
        available = i_idx < t_idx                                # [T, T-1]
        match_prev = (prev.unsqueeze(0) == ids_b.unsqueeze(1))   # [T, T-1]
        mask = (available & match_prev).to(torch.float32)        # [T, T-1]

        # counts[t, v] = number of available transitions at position t where next token == v
        counts = torch.zeros(T, V, device=base_logp.device, dtype=torch.float32)  # [T, V]
        nxt_expanded = nxt.unsqueeze(0).expand(T, -1)            # [T, T-1]
        counts.scatter_add_(1, nxt_expanded, mask)               # [T, V]

        # Number of cache observations per position
        n = counts.sum(dim=1, keepdim=True)                      # [T, 1]
        has_cache = (n > 0).squeeze(1)                           # [T]

        # Adaptive lambda: more observations -> stronger influence
        lam = cache_lambda * n / (n + cache_k)                   # [T, 1]
        log_keep = torch.log1p(-lam)                             # [T, 1]
        log_lam = torch.log(lam)                                 # [T, 1]

        # Cache log probabilities
        cache_logp = torch.log(counts / n + eps)                 # [T, V]

        # Interpolate in log space
        mixed = torch.logaddexp(
            base_logp[b] + log_keep,
            cache_logp + log_lam
        )                                                        # [T, V]

        # Only apply where cache exists
        out[b] = torch.where(has_cache.unsqueeze(1), mixed, base_logp[b])

    return out



class ImprovedGPT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']  # must be 256 (enforced by make_model)
        width = config['width']
        vocab = config['vocab']
        depth = config['depth']
        heads = config['heads']

        # Token embedding only — positional info is handled by RoPE.
        self.token = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList([ImprovedBlock(width, heads) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)

        # Weight tying: output head shares embedding matrix (saves vocab*width params).
        self.head.weight = self.token.weight
        # --- Train-set bigram LM mixture (evaluation only) ---
        import os
        self.bigram_lambda = float(os.environ.get('BIGRAM_LAMBDA', config.get('bigram_lambda', 0.0)))
        self.bigram_beta = float(os.environ.get('BIGRAM_BETA', config.get('bigram_beta', 50.0)))
        self.local_cache_lambda = float(os.environ.get('LOCAL_CACHE_LAMBDA', config.get('local_cache_lambda', 0.0)))
        self.local_cache_k = float(os.environ.get('LOCAL_CACHE_K', config.get('local_cache_k', 2.0)))

        if self.bigram_lambda > 0:
            bigram_logp = _build_bigram_logp(vocab, beta=self.bigram_beta)
            self.register_buffer('bigram_logp', bigram_logp, persistent=True)
        else:
            self.bigram_logp = None

        # Base initialization for all linear/embedding layers.
        self.apply(self._init_weights)

        # Depth-scaled init for residual output projections (GPT-3 / LLaMA trick).
        # Reduces variance growth through the residual stream in deeper networks.
        scale = 0.02 / math.sqrt(2.0 * depth)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=scale)
            nn.init.normal_(block.mlp.w2.weight, std=scale)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.token(ids)  # no positional add — RoPE handles it inside attention
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(ids))

    @torch.no_grad()
    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(ids).float()
        neural_logp = F.log_softmax(logits, dim=-1)

        # Step 1: Bigram mixture (global, from training set)
        if self.bigram_lambda > 0 and self.bigram_logp is not None:
            lam = self.bigram_lambda
            bigram_logp = self.bigram_logp[ids].to(neural_logp.device)
            mixed = torch.logaddexp(
                neural_logp + math.log1p(-lam),
                bigram_logp + math.log(lam),
            )
        else:
            mixed = neural_logp

        # Step 2: Local cache (within current window, causal)
        if self.local_cache_lambda > 0:
            mixed = _apply_local_bigram_cache(
                mixed, ids,
                cache_lambda=self.local_cache_lambda,
                cache_k=self.local_cache_k,
            )

        return mixed


# ---------------------------------------------------------------------------
# Factory (required entry point)
# ---------------------------------------------------------------------------

def build_model(config: dict) -> ImprovedGPT:
    """Construct the improved model.  Called by common.make_model."""
    return ImprovedGPT(config)
