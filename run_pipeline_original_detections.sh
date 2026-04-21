#!/usr/bin/env bash
# run_pipeline_original_detections.sh <bag_dir> [bag_dir2 ...]
#
# Full pipeline specialized for the original detections workflow
#   1. Reuse /detections from the original bag
#   2. Rebuild /tag_measurements_base via solvePnP
#   3. Merge the regenerated topics back into a final bag
#
# Usage:
#   ./run_pipeline_original_detections.sh <bag_dir>
#   KEEP_REPROCESSED=1 ./run_pipeline_original_detections.sh <bag_dir1> <bag_dir2>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEEP_REPROCESSED="${KEEP_REPROCESSED:-0}"
REPROCESS_SCRIPT="$SCRIPT_DIR/reprocess_bag_from_original_detections.sh"
MERGE_SCRIPT="$SCRIPT_DIR/merge_bags.py"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <bag_dir> [bag_dir2 ...]"
    exit 1
fi

for BAG_DIR in "$@"; do
    BAG_DIR="$(realpath "$BAG_DIR")"
    if [ ! -d "$BAG_DIR" ]; then
        echo "[ERROR] Bag directory does not exist: $BAG_DIR"
        exit 1
    fi

    BAG_NAME="$(basename "$BAG_DIR")"
    REPROCESSED="$SCRIPT_DIR/reprocessed_${BAG_NAME}_overlay"
    FINAL="$SCRIPT_DIR/final_${BAG_NAME}_reprocessed_topic_fixed_time"

    echo ""
    echo "========================================================"
    echo "  Processing: $BAG_NAME"
    echo "========================================================"
    echo "  Mode       : original_detections"
    echo "  Reprocessed: $(basename "$REPROCESSED")"
    echo "  Final      : $(basename "$FINAL")"

    echo ""
    echo "[STEP 1/2] Reprocessing from original detections..."
    bash "$REPROCESS_SCRIPT" "$BAG_DIR" "$REPROCESSED"

    echo ""
    echo "[STEP 2/2] Merging with original topics..."
    "$PYTHON_BIN" "$MERGE_SCRIPT" \
        "$BAG_DIR" "$REPROCESSED" "$FINAL"

    if [ "$KEEP_REPROCESSED" = "1" ]; then
        echo ""
        echo "[INFO] Keeping intermediate reprocessed bag for debugging:"
        echo "       $REPROCESSED"
    else
        echo ""
        echo "[INFO] Removing intermediate reprocessed bag..."
        rm -rf "$REPROCESSED"
    fi

    echo ""
    echo "[DONE] Final bag: $FINAL"
done

echo ""
echo "========================================================"
echo "  All bags processed."
echo "========================================================"
