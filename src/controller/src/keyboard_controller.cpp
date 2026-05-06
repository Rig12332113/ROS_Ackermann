#include <chrono>
#include <iostream>
#include <memory>
#include <termios.h>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class KeyboardController: public rclcpp::Node{
public:
    KeyboardController(): 
        Node("keyboard_controller"){
        vel_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/rear_wheel_velocity_controller/commands", 10
        );
        dir_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/front_steering_position_controller/commands", 10
        );
        velocity_ = 0.0;
        direction_ = 0.0;

        velocity_step_ = 0.5;
        direction_step_ = 0.1;

        max_velocity_ = 10.0;
        max_steering_ = 0.6;

        printHelp();
        publishCommand();
    }
    void run(){
        while (rclcpp::ok()) {
            char key = getKey();

            if (key == 'q') {
                velocity_ = 0.0;
                direction_ = 0.0;
                break;
            }

            handleKey(key);
            publishCommand();
            rclcpp::spin_some(this->get_node_base_interface());
        }
    }
    void publishCommand(){
        std_msgs::msg::Float64MultiArray vel_msg;
        vel_msg.data = {velocity_, velocity_};

        std_msgs::msg::Float64MultiArray dir_msg;
        dir_msg.data = {direction_, direction_};

        vel_pub_->publish(vel_msg);
        dir_pub_->publish(dir_msg);

    }

private:
    void printHelp(){
        std::cout << "\nKeyboard Car Controller\n";
        std::cout << "-----------------------\n";
        std::cout << "w : move forward (increase speed) \n";
        std::cout << "s : move backward (decrease speed) \n";
        std::cout << "a : steer left\n";
        std::cout << "d : steer right\n";
    }
    
     double clamp(double value, double min_value, double max_value){
        if (value < min_value) {
        return min_value;
        }
        if (value > max_value) {
        return max_value;
        }
        return value;
    }

    char getKey(){
        char key;

        struct termios old_terminal;
        struct termios new_terminal;

        tcgetattr(STDIN_FILENO, &old_terminal);
        new_terminal = old_terminal;

        new_terminal.c_lflag &= ~(ICANON | ECHO);

        tcsetattr(STDIN_FILENO, TCSANOW, &new_terminal);
        key = getchar();
        tcsetattr(STDIN_FILENO, TCSANOW, &old_terminal);

        return key;
    }
    void handleKey(char key){
        switch (key) {
        case 'w':
            velocity_ += velocity_step_;
            break;

        case 's':
            velocity_ -= velocity_step_;
            break;

        case 'a':
            direction_ += direction_step_;
            break;

        case 'd':
            direction_ -= direction_step_;
            break;
        default:
            break;
        }

        velocity_ = clamp(velocity_, -max_velocity_, max_velocity_);
        direction_ = clamp(direction_, -max_steering_, max_steering_);
    }
        


    double velocity_;
    double direction_;
    double velocity_step_;
    double direction_step_;
    double max_velocity_;
    double max_steering_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr vel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr dir_pub_;
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<KeyboardController>();
  node->run();

  rclcpp::shutdown();
  return 0;
}