from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

import os

def generate_launch_description():
    venvpython = "/home/rick/Desktop/ros2_project/slamNmap/gtsam_venv/lib/python3.12/site-packages"
    old_pythonpath = os.environ.get("PYTHONPATH", "")

    if old_pythonpath:
        pythonpath = venvpython + ":" + old_pythonpath
    else:
        pythonpath = venvpython

    gtsam = Node(
        package="realtime_gtsam",
        executable="gtsam_octo",
        output="screen",
        additional_env={
            "PYTHONPATH": pythonpath,
        },
    )
    compare_script = ExecuteProcess(
        cmd=[
            "python3",
            "/home/rick/Desktop/ros2_project/slamNmap/tool_scripts/compare_odom_paths.py",
        ],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2"
    )
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("car_description"),
                    "launch",
                    "gazebo.launch.py"
                ])
            ])
        ),
        gtsam,
        compare_script,
        rviz
    ])
    