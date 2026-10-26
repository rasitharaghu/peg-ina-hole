#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "samu_control/config.hpp"
#include "samu_control/pid.hpp"
#include "samu_control/foh.hpp"
#include "samu_control/fanuc_dpm_server.hpp"
#include "samu_control/ros_helpers.hpp"
class Node:public rclcpp::Node{
 public:Node():rclcpp::Node("samu_strategy2_ff_pid_foh"){
  auto p=declare_parameter<std::string>("config_file","config/strategy2_ff_pid_foh.json");auto j=load_json(p);cx_=axis_cfg(j,"x");cz_=axis_cfg(j,"z");cr_=axis_cfg(j,"ry");rate_=j.value("control_rate_hz",125.0);sensor_rate_=j.value("sensor_rate_hz",30.0);setup(cx_,px_,fx_);setup(cz_,pz_,fz_);setup(cr_,pr_,fr_);
  auto f=j.at("fanuc");out_=std::make_unique<FanucDpmServer>(f.value("dry_run",true),f.value("host","0.0.0.0"),f.value("port",30000),f.value("map_process_ry_to_fanuc_p",true),f.value("linear_si_to_fanuc",1000.0),f.value("angular_si_to_fanuc",57.295779513));
  fb_=create_subscription<std_msgs::msg::Float64MultiArray>(j.value("feedback_topic","/samu/gocator/feedback_x_z_ry"),10,[this](auto m){meas_=vec3msg(*m);have_fb_=true;update_target();});
  ff_=create_subscription<std_msgs::msg::Float64MultiArray>(j.value("feedforward_topic","/samu/gocator/feedforward_x_z_ry"),10,[this](auto m){ffv_=vec3msg(*m);update_target();});
  timer_=create_wall_timer(std::chrono::duration<double>(1/rate_),[this]{out_->send(fx_.update(),fz_.update(),fr_.update());});
 }
 private:void setup(const AxisCfg&c,PID&p,FOH&f){p.configure(c.kp,c.ki,c.kd,c.lo,c.hi);f.configure(1/rate_,1/sensor_rate_,c.rate,c.accel);} void update_target(){double dt=1/sensor_rate_;Vec3 ufb{};if(have_fb_)ufb={px_.update(cx_.sp,meas_.x,dt),pz_.update(cz_.sp,meas_.z,dt),pr_.update(cr_.sp,meas_.ry,dt)};Vec3 u{cx_.ff_gain*ffv_.x+ufb.x,cz_.ff_gain*ffv_.z+ufb.z,cr_.ff_gain*ffv_.ry+ufb.ry};fx_.setTarget(std::clamp(u.x,cx_.lo,cx_.hi));fz_.setTarget(std::clamp(u.z,cz_.lo,cz_.hi));fr_.setTarget(std::clamp(u.ry,cr_.lo,cr_.hi));}
 AxisCfg cx_,cz_,cr_;PID px_,pz_,pr_;FOH fx_,fz_,fr_;Vec3 meas_{},ffv_{};bool have_fb_{false};double rate_,sensor_rate_;std::unique_ptr<FanucDpmServer>out_;rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr fb_,ff_;rclcpp::TimerBase::SharedPtr timer_;};
int main(int argc,char**argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<Node>());rclcpp::shutdown();}
