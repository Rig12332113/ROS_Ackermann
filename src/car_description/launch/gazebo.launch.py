from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.substitutions import Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
    world_path = PathJoinSubstitution([
        FindPackageShare("car_description"),
        "world",
        "test.sdf"
    ])
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            ])
        ]),
        launch_arguments={
            "gz_args": ["-r ", world_path]
        }.items()
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description,],
        output="screen"
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "robot_description",
            "-name", "ackermann_car",
            "-x", "0",
            "-y", "0",
            "-z", "0.15"
        ],
        output="screen"
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    rear_wheel_velocity_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "rear_wheel_velocity_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    front_steering_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "front_steering_position_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            "config_file": PathJoinSubstitution([
                FindPackageShare("car_description"),
                "config",
                "gazebo_bridge.yaml",
            ])
        }],
        output="screen",
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
        TimerAction(
            period=3.0,
            actions=[
                joint_state_broadcaster_spawner,
            ],
        ),

        TimerAction(
            period=4.0,
            actions=[
                rear_wheel_velocity_controller_spawner,
                front_steering_position_controller_spawner,
            ],
        ),
        bridge
    ])