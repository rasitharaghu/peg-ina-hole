#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "samu_control/config.hpp"
#include "samu_control/pid.hpp"
#include "samu_control/foh.hpp"
#include "samu_control/fanuc_dpm_server.hpp"
#include "samu_control/ros_helpers.hpp"
class Node:public rclcpp::Node{
 public:Node():rclcpp::Node("samu_strategy1_pid_foh"){
  auto p=declare_parameter<std::string>("config_file","config/strategy1_pid_foh.json");auto j=load_json(p);cx_=axis_cfg(j,"x");cz_=axis_cfg(j,"z");cr_=axis_cfg(j,"ry");
  rate_=j.value("control_rate_hz",125.0);sensor_rate_=j.value("sensor_rate_hz",30.0);setup(cx_,px_,fx_);setup(cz_,pz_,fz_);setup(cr_,pr_,fr_);
  auto f=j.at("fanuc");out_=std::make_unique<FanucDpmServer>(f.value("dry_run",true),f.value("host","0.0.0.0"),f.value("port",30000),f.value("map_process_ry_to_fanuc_p",true),f.value("linear_si_to_fanuc",1000.0),f.value("angular_si_to_fanuc",57.295779513));
  sub_=create_subscription<std_msgs::msg::Float64MultiArray>(j.value("feedback_topic","/samu/gocator/feedback_x_z_ry"),10,[this](auto m){auto y=vec3msg(*m);double dt=1.0/sensor_rate_;tx_=px_.update(cx_.sp,y.x,dt);tz_=pz_.update(cz_.sp,y.z,dt);tr_=pr_.update(cr_.sp,y.ry,dt);fx_.setTarget(tx_);fz_.setTarget(tz_);fr_.setTarget(tr_);});
  timer_=create_wall_timer(std::chrono::duration<double>(1.0/rate_),[this]{out_->send(fx_.update(),fz_.update(),fr_.update());});
 }
 private:void setup(const AxisCfg&c,PID&p,FOH&f){p.configure(c.kp,c.ki,c.kd,c.lo,c.hi);f.configure(1/rate_,1/sensor_rate_,c.rate,c.accel);} AxisCfg cx_,cz_,cr_;PID px_,pz_,pr_;FOH fx_,fz_,fr_;double rate_,sensor_rate_,tx_{0},tz_{0},tr_{0};std::unique_ptr<FanucDpmServer>out_;rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;rclcpp::TimerBase::SharedPtr timer_;};
int main(int argc,char**argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<Node>());rclcpp::shutdown();}
