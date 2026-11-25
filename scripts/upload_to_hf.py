"""Upload a collected dataset root (sharded Zarr v3 stores) to a Hugging Face dataset repo.

Uses ``HfApi.upload_folder`` (huggingface_hub>=2), which commits in batches and is resumable: re-running after an
interruption only uploads missing files. Requires ``hf auth login`` first.

    python scripts/upload_to_hf.py --repo-id <user>/point-hjepa-maniskill3-6d \
        --folder outputs/datasets/v1
"""

from __future__ import annotations

import argparse

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--folder", default="outputs/datasets/v1")
    parser.add_argument("--public", action="store_true", help="Create the repo public (default: private).")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    # huggingface_hub>=2 splits large folders into several commits and resumes on re-run.
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.folder,
        ignore_patterns=["**/*.pid", "**/hydra/**", "**/.cache/**"],
        commit_message="Upload Point-H-JEPA ManiSkill3 6D dataset v1",
    )
    print(f"Uploaded {args.folder} -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
