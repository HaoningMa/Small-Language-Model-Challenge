"""Optimized training recipe: EMA + label smoothing + tuned LR schedule.
Default: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--ema-decay', type=float, default=0.999,
                   help='Exponential moving average decay for weights.')
    p.add_argument('--label-smoothing', type=float, default=0.1,
                   help='Label smoothing factor for cross-entropy loss.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)

    # --- EMA: maintain a shadow copy of weights for evaluation ---
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    # --- Track best validation checkpoint ---
    best_bpb = float('inf')
    best_state = None
    best_validation = None
    best_step = None

    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        # Tuned LR: 200-step warmup, cosine decay to 5% of peak (was 100-step / 10%)
        learning_rate = .001 * min(1.,(step+1)/200) * (.05+.95*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = model(batch[:,:-1]).flatten(0,1).float()
            targets = batch[:,1:].flatten()
            # Label smoothing: prevents overconfidence, acts as regularizer
            loss = F.cross_entropy(logits, targets, label_smoothing=args.label_smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()

        # --- EMA update: blend new weights into shadow copy ---
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.mul_(args.ema_decay).add_(p, alpha=1 - args.ema_decay)

        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'lr':learning_rate,
                   'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            intermediate = score(ema_model, *data['validation'], device, 'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step': step + 1, **intermediate})
            print(json.dumps({'validation': validation_history[-1]}), flush=True)
            # --- Save best EMA weights ---
            if intermediate['bpb'] < best_bpb:
                best_bpb = intermediate['bpb']
                best_step = step + 1
                best_validation = dict(intermediate)
                best_state = {k: v.detach().cpu().clone()
                              for k, v in ema_model.state_dict().items()}

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds

    # --- Final validation uses EMA model ---
    validation = score(ema_model, *data['validation'], device, 'fp32')
    validation.pop('window_nll_nats')
    # --- Compare final step with best, load best ---
    if validation['bpb'] < best_bpb:
        best_bpb = validation['bpb']
        best_step = args.steps
        best_validation = dict(validation)
        best_state = {k: v.detach().cpu().clone()
                      for k, v in ema_model.state_dict().items()}
    if best_state is not None:
        ema_model.cpu()
        ema_model.load_state_dict(best_state)
        validation = best_validation
        print(f'[best checkpoint] step={best_step}, validation_bpb={best_bpb:.6f}', flush=True)
    else:
        ema_model.cpu()

    checkpoint = args.run_dir/'checkpoint.pt'
    # Save EMA weights — this is what gets evaluated and submitted
    torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                'model': ema_model.state_dict(), 'seed': args.seed,
                'train_tokens': args.steps * args.batch_size * 256,
                'ema_decay': args.ema_decay, 'label_smoothing': args.label_smoothing,
                'selected_step': best_step}, checkpoint)

    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              'ema_decay':args.ema_decay,'label_smoothing':args.label_smoothing,
              'selected_step': best_step,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
