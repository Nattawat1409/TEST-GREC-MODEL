# Helpers sourced by models/*/setup.sh and run.sh.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/third_party" "$ROOT/checkpoints" "$ROOT/outputs"

# clone_at <url> <dir> <commit>: clone once, then pin to the commit the patches were made against
clone_at() {
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" fetch -q origin "$3" 2>/dev/null || true
  git -C "$2" checkout -q "$3"
}

# apply_patch <repo> <patch>: apply once (safe to re-run)
apply_patch() {
  if git -C "$1" apply --reverse --check "$2" 2>/dev/null; then
    echo "patch already applied: $(basename "$2")"
  else
    git -C "$1" apply "$2" && echo "patch applied: $(basename "$2")"
  fi
}

# need_file <path> <where to get it>: stop with a clear message if a manual download is missing
need_file() {
  if [ ! -e "$1" ]; then
    echo ""
    echo "Missing: $1"
    echo "Download it from: $2"
    echo "then run this script again."
    exit 1
  fi
}

# check_data: the test annotations must exist before any model runs
check_data() {
  need_file "$ROOT/data/annotations/optsar_test.json" \
    "python scripts/download_optsar.py && python scripts/prepare_optsar_test.py"
}
