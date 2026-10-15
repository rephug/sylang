"""Score a saved design-v3 checkpoint on freshly built panels, optionally with fixed-shape
(letter-digit-letter) identifiers, so that corpus variants trained without variable-length
identifiers can be judged on their own identifier distribution.

    python -I eval3.py --source <packet> --checkpoint <study3 .pt> [--lengths v2|v3] [--per-k 16] \
        --out <report.json>

Panels (test-split identifiers, seed 7 as in ``study3.panels``): ``iid`` (K 1..12, trained wording,
trained combinations), ``wording`` (held-out wording), ``combo`` (a held-out combination), ``long``
(K 13..16), ``full`` (held-out wording + held-out combination + K 13..16), ``reorder``, ``swap``,
``flip``. With ``--lengths v2`` every identifier is a letter-digit-letter string from the v2 test
split; with ``v3`` the panels are exactly those of the study. Only aggregates are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import language as L  # noqa: E402
import study3  # noqa: E402
from followup import Runtime  # noqa: E402


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--arm', help='rendering arm; default: parsed from the checkpoint name')
    parser.add_argument('--lengths', default='v2', choices=('v2', 'v3'))
    parser.add_argument('--per-k', type=int, default=16)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rt = Runtime(args.source)
    renderer = L.Renderer(rt)
    name = Path(args.checkpoint).name
    arm = args.arm or name.split('-')[0]
    model = rt.load(rt.build(0), args.checkpoint).eval()
    if '-alibi' in name:
        L.use_alibi(model, rt.recipe['heads'], rt.tiny.PAD)
    lengths = None if args.lengths == 'v2' else study3.ID_LENGTHS
    raw = study3.panels(study3.K_TRAIN, args.per_k, lengths=lengths)
    raw = {k: v for k, v in raw.items() if not k.startswith('dev')}
    panels = {panel: [dict(prompt=study3.render(renderer, arm, it), target=renderer.target(list(it)), statements=list(it))
                      for it in items] for panel, items in raw.items()}
    scores = L.score(rt, model, panels)
    report = {'checkpoint': name, 'arm': arm, 'lengths': args.lengths, 'per_k': args.per_k,
              'panels': {p: {k: v for k, v in s.items() if k != 'flags'} for p, s in scores.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    for p in ('iid', 'wording', 'combo', 'long', 'full', 'reorder', 'swap', 'flip'):
        s = scores[p]
        print(f"{p:8s} exact {100 * s['exact']:5.1f}  selected {100 * (s['selected'] or 0):5.1f}  n {s['n']}  "
              f"by_k {s['by_k']}")


if __name__ == '__main__':
    main()
