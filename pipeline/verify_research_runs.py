"""Verify deterministic restart against an uninterrupted two-batch smoke run."""
import argparse
import json
from pathlib import Path
import torch


def verify(resumed, continuous):
    a = torch.load(resumed / 'last.pt', map_location='cpu', weights_only=False)
    b = torch.load(continuous / 'last.pt', map_location='cpu', weights_only=False)
    if not a['config']['smoke_only'] or not b['config']['smoke_only']:
        raise ValueError('This check expects isolated smoke checkpoints')
    checks = {'same_epoch': a['epoch'] == b['epoch'],
              'same_validation_loss': a['validation_loss'] == b['validation_loss'],
              'same_model_tensors': all(torch.equal(v, b['model'][k]) for k, v in a['model'].items()),
              'same_ema_tensors': all(torch.equal(v, b['ema'][k]) for k, v in a['ema'].items()),
              'same_loader_rng': torch.equal(a['rng']['loader'], b['rng']['loader'])}
    result = dict(checks=checks, passed=all(checks.values()), resumed=str(resumed), continuous=str(continuous))
    (resumed / 'resume_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise AssertionError('Resume differs from uninterrupted training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resumed', type=Path)
    parser.add_argument('continuous', type=Path)
    args = parser.parse_args()
    verify(args.resumed, args.continuous)
