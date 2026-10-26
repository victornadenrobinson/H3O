#!/bin/bash
# Send the newest data bundle to Young after checking the build log passed. Run on Perlmutter from the repo root.
# usage: bash mlp_pipeline/send_bundle.sh [--dry-run] [--force]   (--force: skip the runs/build.log check, e.g. when build_all.sh ran before it wrote that log)
# env overrides: YOUNG=mmm0037@young-ng.rc.ucl.ac.uk  YOUNG_DIR=/lustre/scratch/mmm0037/H3O  YOUNG_KEY=~/.ssh/id_young
set -euo pipefail
cd "$(dirname "$0")/.."
YOUNG=${YOUNG:-mmm0037@young-ng.rc.ucl.ac.uk}   # login confirmed 2026-10-25 (login22.young.ucl.ac.uk)
YOUNG_DIR=${YOUNG_DIR:-/lustre/scratch/mmm0037/H3O}
LOG=runs/build.log; DRY=0; FORCE=0
for arg in "$@"; do [ "$arg" = "--dry-run" ] && DRY=1; [ "$arg" = "--force" ] && FORCE=1; done
if [ "$FORCE" = 0 ]; then
  [ -f "$LOG" ] || { echo "no $LOG: run build_all.sh first (or pass --force after checking the PASS lines yourself)"; exit 1; }
  if grep -q "FAIL:" "$LOG" || ! grep -q "^PASS:" "$LOG"; then
    echo "build log has FAIL lines or no PASS summary; not sending. Lines:"; grep -E "FAIL|PASS" "$LOG" | head; exit 1
  fi
fi
B=$(ls -t runs/H3O_bundle_*.tar.gz 2>/dev/null | head -1); [ -n "$B" ] || { echo "no bundle in runs/"; exit 1; }
SUM=$(sha256sum "$B" | cut -d' ' -f1)
echo "bundle: $B ($(du -h "$B" | cut -f1)), sha256 $SUM"
if [ "$DRY" = 1 ]; then echo "dry run: would upload to $YOUNG:$YOUNG_DIR/"; exit 0; fi
# Upload via `ssh ... cat >` (not scp/sftp): a chatty remote ~/.bashrc corrupts scp's protocol stream but cannot touch data on stdin.
KEY=${YOUNG_KEY:-$HOME/.ssh/id_young}; SSH="ssh -x -o BatchMode=no"; [ -f "$KEY" ] && SSH="$SSH -i $KEY"
NAME=$(basename "$B")
$SSH "$YOUNG" "mkdir -p $YOUNG_DIR && cat > $YOUNG_DIR/$NAME.part && mv $YOUNG_DIR/$NAME.part $YOUNG_DIR/$NAME" < "$B" >/dev/null
REMOTE=$($SSH "$YOUNG" "sha256sum $YOUNG_DIR/$NAME" 2>/dev/null | grep -Eo '^[0-9a-f]{64}' | head -1)
[ "$REMOTE" = "$SUM" ] || { echo "CHECKSUM MISMATCH: local $SUM remote ${REMOTE:-<none>}"; exit 1; }
echo "uploaded, checksum verified. Unpacking..."
$SSH "$YOUNG" "cd $YOUNG_DIR && tar xzf $NAME && ls runs | head -30" 2>/dev/null | grep -v -E "^(OK[0-9]|bashrc)"
echo "sent and verified on Young."
