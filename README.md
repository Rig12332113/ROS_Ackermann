# 1. build ROS file
```
colcon build
source install/setup.bash
```
# 2. run gazebo simulation
```
ros2 launch car_description gazebo.launch.py 
```
# 3. run ros2_control 
```
ros2 topic pub /rear_wheel_velocity_controller/commands std_msgs/msg/Float64MultiArray "{data: [2.0, 2.0]}"
ros2 topic pub /front_steering_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.3, 0.3]}"
```
