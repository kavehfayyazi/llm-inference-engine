"""CLI: generate text from a prompt."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import EngineConfig
from engine.generate import generate
from engine.model import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    cfg = EngineConfig()
    if args.model_id:
        cfg.model_id = args.model_id
    lm = load(cfg)

    _, text = generate(lm, args.prompt, max_new_tokens=args.max_new_tokens)
    print(text)


if __name__ == "__main__":
    main()
