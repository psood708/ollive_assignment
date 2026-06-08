#!/bin/bash
set -e

SPACE_URL="${1:-https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME}"
TMPDIR=$(mktemp -d)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Building Space from $SCRIPT_DIR"
echo "Deploying to $SPACE_URL"

cp -r "$SCRIPT_DIR/shared"           "$TMPDIR/shared"
cp -r "$SCRIPT_DIR/oss_assistant"    "$TMPDIR/oss_assistant"
cp    "$SCRIPT_DIR/requirements.txt" "$TMPDIR/requirements.txt"
cp    "$SCRIPT_DIR/oss_assistant/Dockerfile" "$TMPDIR/Dockerfile"
cp    "$SCRIPT_DIR/space_readme.md"  "$TMPDIR/README.md"

cd "$TMPDIR"
git init -q
git add .
git commit -q -m "deploy: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git remote add space "$SPACE_URL"
git push space master:main --force

rm -rf "$TMPDIR"
echo "Done. Space is rebuilding at $SPACE_URL"
