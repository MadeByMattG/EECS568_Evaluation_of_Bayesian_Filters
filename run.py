import argparse
import os

import rclpy
from system.RobotSystem import RobotSystem
from world.world2d import world2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag-name",
        default=os.getenv("ROB530_BAG_NAME", ""),
        help="Bag name used for saved evaluation artifacts.",
    )
    parser.add_argument(
        "--eval-dir",
        default=os.getenv("ROB530_EVAL_DIR", "eval_outputs"),
        help="Directory for saved evaluation artifacts.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show Matplotlib windows instead of headless save-only mode.",
    )
    args = parser.parse_args()

    if args.bag_name:
        os.environ["ROB530_BAG_NAME"] = args.bag_name
    if args.eval_dir:
        os.environ["ROB530_EVAL_DIR"] = args.eval_dir
    if args.show_plots:
        os.environ["ROB530_NO_SHOW"] = "0"
    else:
        os.environ.setdefault("ROB530_NO_SHOW", "1")

    rclpy.init()
    world = world2d()
    robot_system = RobotSystem(world)
    try:
        rclpy.spin(robot_system)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            robot_system.plot_results()
        except KeyboardInterrupt:
            # Allow Ctrl-C to close plot display without an ugly traceback.
            pass
        finally:
            robot_system.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
