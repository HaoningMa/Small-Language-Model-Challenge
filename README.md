# Small-Language-Model-Challenge
# DASE7506 MP1: Small Language Model Challenge

Improving bits-per-byte (BPB) on WikiText-2 through architecture, training, and inference-time optimizations.

## Final Result

| Model | Test BPB | Eval Time | Params |
|---|---|---|---|
| Baseline | 2.1013 | 11.1s | 1.09M |
| **Our Model** | **1.5677** | 50.6s | 6.85M |

25.4% BPB reduction from baseline. All hard constraints satisfied (eval time ≤ 5× baseline, memory ≤ 4 GiB, checkpoint ≤ 64 MiB).

## Key Techniques

- **Architecture**: RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU gated MLP
- **Training**: EMA weight averaging (decay=0.995), best-checkpoint selection, label smoothing ablation (ls=0 optimal)
- **Inference**: Training-set bigram mixture (λ=0.08), fully-vectorized causal local cache (λ=0.30, k=2.0)
- **Scale**: 256 width × 8 depth Transformer, 4000 training steps, best checkpoint at step 2500

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

## AI Usage Disclosure

This project benefited from AI-assisted development tools during the development process, including code structure suggestions, debugging assistance, hyperparameter exploration ideas, and report writing support. All final implementation, experimental design, and results were independently reviewed, validated, and confirmed by the author. The codebase is fully reproducible from the provided configuration and training scripts.

