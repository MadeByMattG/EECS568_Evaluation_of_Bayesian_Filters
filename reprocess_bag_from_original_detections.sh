#!/usr/bin/env bash
# reprocess_bag_from_original_detections.sh <bag_dir> [output_bag_name]
#
# Rebuild /tag_measurements_base from the ORIGINAL /detections topic already
# present in the bag, instead of rerunning apriltag_ros on the recorded images.
#
# This is useful when the original live detector saw tags that cannot be
# recovered reliably from the recorded image stream (for example if frames were
# dropped or recorded at a lower effective quality than the detector saw live).
#
# Outputs a "reprocessed" bag that still fits the existing merge_bags.py flow:
#   - /image_rect               (pass-through from original bag)
#   - /detections_reprocessed   (relay of original /detections)
#   - /tag_measurements_base    (computed from original detections via solvePnP)
#   - /camera/camera_info       (pass-through from original bag)
#   - /tf_static                (base_link -> camera static transform)

set -e

BAG_DIR="${1:?Usage: $0 <bag_dir> [output_bag_name]}"
OUTPUT_BAG="${2:-reprocessed_$(basename "$BAG_DIR")}"

BAG_DIR="$(realpath "$BAG_DIR")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPROCESSED_DETECTIONS_TOPIC="/detections_reprocessed"
TAG_MEASUREMENTS_TOPIC="/tag_measurements_base"

echo "=== ROB530 Original-Detection Reprocessor ==="
echo "  Input  : $BAG_DIR"
echo "  Output : $OUTPUT_BAG"
echo "  Source : /detections (original bag)"
echo "  Relay  : $REPROCESSED_DETECTIONS_TOPIC"
echo "  Measure: $TAG_MEASUREMENTS_TOPIC"
echo ""

source /opt/ros/jazzy/setup.bash

if [ -d "$OUTPUT_BAG" ]; then
    echo "[WARN] Output folder '$OUTPUT_BAG' already exists - deleting it."
    rm -rf "$OUTPUT_BAG"
fi

PIDS=()

cleanup() {
    echo ""
    echo "[INFO] Shutting down background processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "[INFO] Done. Output bag: $OUTPUT_BAG"
}
trap cleanup EXIT

echo "[INFO] Publishing static transform base_link -> camera..."
ros2 run tf2_ros static_transform_publisher \
    --x 0.05751 --y 0 --z 0.02936 \
    --roll -1.57 --pitch 0 --yaw -1.57 \
    --frame-id base_link --child-frame-id camera \
    --ros-args -p use_sim_time:=true \
    &
PIDS+=($!)
sleep 1

echo "[INFO] Starting bag recorder -> $OUTPUT_BAG"
ros2 bag record -o "$OUTPUT_BAG" \
    /image_rect \
    "$REPROCESSED_DETECTIONS_TOPIC" \
    "$TAG_MEASUREMENTS_TOPIC" \
    /camera/camera_info \
    /tf_static \
    &
PIDS+=($!)
sleep 1

echo "[INFO] Starting detection relay node..."
python3 "$SCRIPT_DIR/detection_relay_node.py" --ros-args \
    -p input_topic:=/detections \
    -p output_topic:="$REPROCESSED_DETECTIONS_TOPIC" \
    &
PIDS+=($!)
sleep 1

echo "[INFO] Starting tag measurement node (solvePnP on original detections)..."
python3 "$SCRIPT_DIR/tag_measurement_node.py" --ros-args \
    -p tag_size:=0.077 \
    &
PIDS+=($!)
sleep 1

echo "[INFO] Playing bag... (this will block until the bag ends)"
ros2 bag play "$BAG_DIR" \
    --clock \
    --rate 0.5 \
    --exclude-topics \
    /tf_static \
    --start-paused
