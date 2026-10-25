#!/bin/bash
# Send the newest data bundle to Young after checking the build log passed. Run on Perlmutter from the repo root.
# usage: bash mlp_pipeline/send_bundle.sh [--dry-run]
# env overrides: YOUNG=mmm0037@young.rc.ucl.ac.uk  YOUNG_DIR=/lustre/scratch/mmm0037/H3O   (hostname is a guess: use the one you ssh to)
set -euo pipefail
cd "$(dirname "$0")/.."
YOUNG=${YOUNG:-mmm0037@young.rc.ucl.ac.uk}
YOUNG_DIR=${YOUNG_DIR:-/lustre/scratch/mmm0037/H3O}
LOG=runs/build.log
[ -f "$LOG" ] || { echo "no $LOG: run build_all.sh first"; exit 1; }
if grep -q "FAIL:" "$LOG" || ! grep -q "^PASS:" "$LOG"; then
  echo "build log has FAIL lines or no PASS summary; not sending. Lines:"; grep -E "FAIL|PASS" "$LOG" | head; exit 1
fi
B=$(ls -t runs/H3O_bundle_*.tar.gz 2>/dev/null | head -1); [ -n "$B" ] || { echo "no bundle in runs/"; exit 1; }
SUM=$(sha256sum "$B" | cut -d' ' -f1)
echo "bundle: $B ($(du -h "$B" | cut -f1)), sha256 $SUM"
if [ "${1:-}" = "--dry-run" ]; then echo "dry run: would scp to $YOUNG:$YOUNG_DIR/"; exit 0; fi
scp "$B" "$YOUNG:$YOUNG_DIR/"
ssh "$YOUNG" "cd $YOUNG_DIR && echo '$SUM  $(basename "$B")' | sha256sum -c - && tar xzf $(basename "$B") && ls -la runs | head -20"
echo "sent and verified on Young."
