#!/bin/bash
# Build benchmark binaries for multiple commits
# Run this, then let laptop cool, then run bench_run.sh

set -e

COMMITS=("main" "main~1" "main~2")
LABELS=("main" "main~1" "main~2")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TMP_DIR="/tmp/bench_compare"

mkdir -p "$TMP_DIR"

cd "$PROJECT_DIR"

# Save current state
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD)
STASH_NEEDED=false
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Stashing uncommitted changes..."
    git stash push -m "bench_build temp stash"
    STASH_NEEDED=true
fi

cleanup() {
    cd "$PROJECT_DIR"
    git checkout "$ORIGINAL_BRANCH" --quiet 2>/dev/null || true
    if $STASH_NEEDED; then
        echo "Restoring stashed changes..."
        git stash pop --quiet || true
    fi
}
trap cleanup EXIT

echo "=== Building ${#COMMITS[@]} versions ==="
echo "Output: $TMP_DIR"
echo ""

for i in "${!COMMITS[@]}"; do
    commit="${COMMITS[$i]}"
    label="${LABELS[$i]}"
    sha=$(git rev-parse --short "$commit")
    echo "[$((i+1))/${#COMMITS[@]}] Building $label ($sha)..."
    git checkout "$commit" --quiet
    cargo build --release --bin bench_voronoi 2>&1 | grep -E "(Compiling|Finished)" || true
    cp target/release/bench_voronoi "$TMP_DIR/bench_$i"
done

echo ""
echo "=== Build complete ==="
echo "Binaries:"
for i in "${!COMMITS[@]}"; do
    echo "  $TMP_DIR/bench_$i  (${LABELS[$i]})"
done
echo ""
echo "Now let your laptop cool down, then run:"
echo "  ./scripts/bench_run.sh"
