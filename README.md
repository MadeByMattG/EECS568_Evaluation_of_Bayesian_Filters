# ROB530 Final Project Filter Comparison

This repository compares `EKF`, `UKF`, `PF`, and `InEKF` for 2D robot localization in ROS 2 using odometry and AprilTag-based landmark observations.

The current workflow is:

1. Record a bag on the robot.
2. Rebuild AprilTag measurement topics from the recorded detections into an intermediate reprocessed bag.
3. Merge the regenerated topics into a final evaluation bag.
4. Run one of the filters with `run.py`.
5. Save plots and metrics to `eval_outputs/`.

## Environment

This repo is currently set up for:

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3

Python packages used by the project include:

- `numpy`
- `scipy`
- `pyyaml`
- `matplotlib`
- `rosbags`

If you use the optional aligned trajectory metrics, also install:

- `evo`

Typical setup:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
```

## Main Files

- run.py: starts the estimator node
- system/RobotSystem.py: ROS callbacks, gating, evaluation, and filter orchestration
- config/settings.yaml: runtime configuration
- filter/EKF.py
- filter/UKF.py
- filter/PF.py
- filter/InEKF.py
- detection_relay_node.py: relays `/detections` to `/detections_reprocessed`
- tag_measurement_node.py: converts `/detections_reprocessed` into `/tag_measurements_base`
- reprocess_bag_from_original_detections.sh: rebuilds tag topics from an original bag
- run_pipeline_original_detections.sh: reprocess + merge pipeline
- merge_bags.py: merges original and regenerated topics into the final bag

## Topics Used by the Estimator

Current defaults in config/settings.yaml:

- Prediction topic: `/odom`
- Correction topic: `/tag_measurements_base`
- Alternate detection topic: `/detections_reprocessed`
- Ground-truth topic: `/ground_truth/mbot_corrected`

Raw-to-processed AprilTag topic flow:

```text
/detections
  -> detection_relay_node.py
  -> /detections_reprocessed
  -> tag_measurement_node.py
  -> /tag_measurements_base
```

Topics in order:

1. `/detections`: raw AprilTag detections from the original bag or live system
2. `/detections_reprocessed`: relayed/reprocessed AprilTag detection topic
3. `/tag_measurements_base`: base-frame tag measurement topic consumed by the estimator correction step

## Recording a New Bag on the Robot

Use `ros2 bag record` on the robot to capture the topics needed for this pipeline. A practical command is:

```bash
ros2 bag record -o my_run \
  /odom \
  /detections \
  /image_rect \
  /camera/camera_info \
  /tf \
  /tf_static \
  /ground_truth/mbot_corrected
```

Notes:

- `/detections` is the important raw AprilTag topic before reprocessing.
- `/ground_truth/mbot_corrected` is only needed if you want evaluation against mocap.
- If your robot publishes different topic names, update the command and then update config/settings.yaml to match.

## Reprocessing and Final Bag Creation

If you recorded `/detections` and want to rebuild the tag measurement topic from those original detections:

```bash
./run_pipeline_original_detections.sh /path/to/original_bag
```

This does two things:

1. Rebuilds an intermediate reprocessed bag containing:
   - `/detections_reprocessed`
   - `/tag_measurements_base`
   - `/camera/camera_info`
   - `/tf_static`
2. Merges that intermediate reprocessed bag with the original bag into a final bag named like:
   - `final_<bag_name>_reprocessed_topic_fixed_time`

If you want to keep the intermediate reprocessed bag:

```bash
KEEP_REPROCESSED=1 ./run_pipeline_original_detections.sh /path/to/original_bag
```

If you only want the reprocessing step without the final merge:

```bash
./reprocess_bag_from_original_detections.sh /path/to/original_bag
```

## Running the Estimator

Set the filter in config/settings.yaml:

```yaml
filter_name: EKF
```

Supported values:

- `EKF`
- `UKF`
- `PF`
- `InEKF`

Then start the estimator:

```bash
python3 run.py --bag-name final_rosbag2_2026_04_10-13_34_37_core_reprocessed_topic_fixed_time
```

This writes evaluation artifacts to `eval_outputs/` and saves:

- summary text
- metrics JSON
- error plots

To show Matplotlib windows interactively:

```bash
python3 run.py --bag-name final_rosbag2_2026_04_10-13_34_37_core_reprocessed_topic_fixed_time --show-plots
```

## Playing a Bag While Running a Filter

Terminal 1:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python3 run.py --bag-name final_rosbag2_2026_04_10-13_34_37_core_reprocessed_topic_fixed_time
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
ros2 bag play final_rosbag2_2026_04_10-13_34_37_core_reprocessed_topic_fixed_time --clock --rate 0.5
```

For the final comparison runs in this project, all filters were typically played at `0.5x` so that `PF` could keep up more reliably.

## Example Run Commands

These are the usual commands for comparing all four filters on the same bag. Change `filter_name` in config/settings.yaml before each run.

Estimator:

```bash
python3 run.py --bag-name final_rosbag2_2026_04_10-13_34_37_core_reprocessed_topic_fixed_time
```

Bag playback:

```bash
ros2 bag play final_rosbag2_2026_04_10-13_34_37_core_reprocessed_topic_fixed_time --clock --rate 0.5
```

## Evaluation Outputs

Each run saves results under `eval_outputs/<bag>_<filter>/`.

Current metrics include:

- position RMSE
- heading RMSE
- full 3-DOF chi-square pass rate
- split consistency metrics:
  - position chi-square pass rate
  - heading chi-square pass rate
- 3-sigma coverage rates
- runtime and CPU usage
- optional `evo`-aligned position metrics if `evo` is installed

Ground truth is used for evaluation only. It is not fed back into the filter correction step.

## Notes on the Software Design

- Prediction is callback-driven by `/odom`.
- Correction is callback-driven by `/tag_measurements_base` or `/detections_reprocessed`.
- AprilTag updates are gated before correction using:
  - detector quality
  - bearing/range innovation limits
  - world-frame consistency checks
- The same ROS callback and evaluation structure is shared across all filters; only the filter internals differ.

## Configuration

All main settings live in config/settings.yaml, including:

- active filter
- motion noise
- measurement noise
- PF-specific settings
- UKF-specific settings
- mocap alignment parameters
- measurement gating thresholds
- topic names

## Quick Start

The rosbag used for final evaluation in this project can be found at the following link:
https://drive.google.com/file/d/1XaHmpG6yd947EOEoRxFWjYKZs1DBgRw4/view?usp=sharing

If you already have a final bag and just want to run a filter:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate

# edit config/settings.yaml and choose EKF / UKF / PF / InEKF
python3 run.py --bag-name final_rosbag2_2026_04_10-13_34_37_core_reprocessed_topic_fixed_time
```

In another terminal:

```bash
source /opt/ros/jazzy/setup.bash
ros2 bag play final_rosbag2_2026_04_10-13_34_37_core_reprocessed_topic_fixed_time --clock --rate 0.5
```
