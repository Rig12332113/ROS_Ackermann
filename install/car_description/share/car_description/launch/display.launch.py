from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    robot_description_path = PathJoinSubstitution([
        FindPackageShare("car_description"),
        "urdf",
        "car.urdf.xacro"
    ])

    robot_description = {
        "robot_description": ParameterValue(
            Command([
                "xacro ",
                robot_description_path
            ]),
            value_type=str
        )
    }

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[robot_description],
            output="screen",
        ),

        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            output="screen",
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
        ),
    ])