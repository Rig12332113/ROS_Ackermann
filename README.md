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
<img width="2560" height="1600" alt="Screenshot from 2026-05-02 21-30-34" src="https://github.com/user-attachments/assets/8e833e6f-f8ec-4eec-8abb-91962f18b14e" />

