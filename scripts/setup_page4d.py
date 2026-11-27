"""Fetch PAGE-4D code + weights for the per-level point-cloud decoders.

- Clones https://github.com/kaichen-z/PAGE4D into ``third_party/PAGE4D`` pinned at ``c4a39ac``
  (the commit JEPA4R's decoder was built against).
- Downloads ``checkpoint_nomask.pt`` (6,782,314,015 bytes) from the HF dataset ``zhouk777/PAGE4D`` into
  ``checkpoints/page4d/`` (resumable ``.part`` file, size-checked).

Usage: ``python scripts/setup_page4d.py [--root .] [--skip-code] [--skip-weights]``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

PAGE4D_REPO = "https://github.com/kaichen-z/PAGE4D"
PAGE4D_COMMIT = "c4a39ac"
CKPT_URL = "https://huggingface.co/datasets/zhouk777/PAGE4D/resolve/main/checkpoint_nomask.pt"
CKPT_SIZE = 6_782_314_015


def setup_code(dest: Path) -> None:
    if not (dest / ".git").is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", PAGE4D_REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", PAGE4D_COMMIT], check=True)
    print(f"[page4d] code at {dest} @ {PAGE4D_COMMIT}")


def setup_weights(dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size == CKPT_SIZE:
        print(f"[page4d] weights present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(CKPT_URL, headers={"User-Agent": "point-hjepa-setup"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req) as resp:
        if have and resp.status != 206:  # server ignored the range: restart
            have = 0
        mode = "ab" if have else "wb"
        with open(part, mode) as f:
            done, last = have, -1
            while chunk := resp.read(1 << 22):
                f.write(chunk)
                done += len(chunk)
                pct = int(100 * done / CKPT_SIZE)
                if pct != last and pct % 5 == 0:
                    print(f"[page4d] {done / 1e9:.2f} / {CKPT_SIZE / 1e9:.2f} GB ({pct}%)", flush=True)
                    last = pct
    size = part.stat().st_size
    if size != CKPT_SIZE:
        raise RuntimeError(f"Downloaded {size} bytes, expected {CKPT_SIZE}; re-run to resume")
    part.rename(dest)
    print(f"[page4d] weights at {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--skip-code", action="store_true")
    ap.add_argument("--skip-weights", action="store_true")
    args = ap.parse_args()
    if not args.skip_code:
        setup_code(args.root / "third_party" / "PAGE4D")
    if not args.skip_weights:
        setup_weights(args.root / "checkpoints" / "page4d" / "checkpoint_nomask.pt")
    return None


if __name__ == "__main__":
    sys.exit(main())
