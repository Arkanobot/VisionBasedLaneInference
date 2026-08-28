#!/usr/bin/env bash
# Upload deploy/build/ to a Hugging Face Space.
#
#   ./deploy/push.sh <your-username>/<space-name>
#
# Uses huggingface_hub over HTTPS rather than git, because the 13.5 MB
# checkpoint needs Git LFS and installing that would touch the system.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${1:-}"
[ -n "$REPO" ] || { echo "usage: ./deploy/push.sh <username>/<space-name>"; exit 1; }

[ -d deploy/build ] || ./deploy/make_bundle.sh

.venv/bin/python - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi

repo = sys.argv[1]
api = HfApi()
who = api.whoami()["name"]
print(f"authenticated as {who}")

api.create_repo(repo_id=repo, repo_type="space", space_sdk="docker",
                private=True, exist_ok=True)
print(f"space ready: {repo} (private)")

api.upload_folder(folder_path="deploy/build", repo_id=repo, repo_type="space",
                  commit_message="Deploy lane inference app")
print(f"\n  https://huggingface.co/spaces/{repo}")
print("  first build takes ~5 minutes; watch the Logs tab")
PY
