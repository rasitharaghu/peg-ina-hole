#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <deque>
#include <cmath>
#include "samu_control/config.hpp"
#include "samu_control/pid.hpp"
#include "samu_control/foh.hpp"
#include "samu_control/fanuc_dpm_server.hpp"
#include "samu_control/ros_helpers.hpp"
class Node:public rclcpp::Node{
 public:Node():rclcpp::Node("samu_strategy3_preview_buffer"){
  auto p=declare_parameter<std::string>("config_file","config/strategy3_preview_buffer.json");j_=load_json(p);cx_=axis_cfg(j_,"x");cz_=axis_cfg(j_,"z");cr_=axis_cfg(j_,"ry");rate_=j_.value("control_rate_hz",125.0);sensor_rate_=j_.value("sensor_rate_hz",30.0);setup(cx_,px_,fx_);setup(cz_,pz_,fz_);setup(cr_,pr_,fr_);
  auto b=j_.at("buffer");max_samples_=b.value("max_samples",2000);match_tol_=b.value("spatial_match_tolerance_m",0.005);max_age_=b.value("max_age_s",10.0);min_conf_=b.value("min_confidence",0.5);mode_=b.value("mode","spatial_buffer");delay_=b.value("fixed_delay_s",0.0);
  auto f=j_.at("fanuc");out_=std::make_unique<FanucDpmServer>(f.value("dry_run",true),f.value("host","0.0.0.0"),f.value("port",30000),f.value("map_process_ry_to_fanuc_p",true),f.value("linear_si_to_fanuc",1000.0),f.value("angular_si_to_fanuc",57.295779513));
  // Surface sample layout: [px,py,pz, measured_x,measured_z,measured_ry, ff_x,ff_z,ff_ry, confidence]
  surf_=create_subscription<std_msgs::msg::Float64MultiArray>(j_.value("surface_topic","/samu/gocator/surface_sample"),50,[this](auto m){if(m->data.size()<10)return;SurfaceSample s; s.stamp=now().seconds();s.px=m->data[0];s.py=m->data[1];s.pz=m->data[2];s.measured={m->data[3],m->data[4],m->data[5]};s.feedforward={m->data[6],m->data[7],m->data[8]};s.confidence=m->data[9];buf_.push_back(s);while(buf_.size()>max_samples_)buf_.pop_front();});
  // Current dispenser TCP layout in SAME common/workpiece frame: [x,y,z,ry]
  tcp_=create_subscription<std_msgs::msg::Float64MultiArray>(j_.value("tcp_topic","/samu/dispenser/tcp_pose"),20,[this](auto m){if(m->data.size()>=3){tcp_x_=m->data[0];tcp_y_=m->data[1];tcp_z_=m->data[2];have_tcp_=true;}});
  timer_=create_wall_timer(std::chrono::duration<double>(1/rate_),[this]{step();});
 }
 private:
  void setup(const AxisCfg&c,PID&p,FOH&f){p.configure(c.kp,c.ki,c.kd,c.lo,c.hi);f.configure(1/rate_,1/sensor_rate_,c.rate,c.accel);}
  double dist(const SurfaceSample&s)const{return std::sqrt((s.px-tcp_x_)*(s.px-tcp_x_)+(s.py-tcp_y_)*(s.py-tcp_y_)+(s.pz-tcp_z_)*(s.pz-tcp_z_));}
  bool select(SurfaceSample&s){if(buf_.empty())return false;const double t=now().seconds();
    while(!buf_.empty() && t-buf_.front().stamp>max_age_)buf_.pop_front(); if(buf_.empty())return false;
    if(mode_=="fixed_delay"){double best=1e9;bool ok=false;for(auto&v:buf_){double d=std::abs((t-v.stamp)-delay_);if(d<best){best=d;s=v;ok=true;}}return ok&&s.confidence>=min_conf_;}
    if(!have_tcp_)return false;double best=1e9;bool ok=false;for(auto&v:buf_){double d=dist(v);if(d<best){best=d;s=v;ok=true;}}return ok&&best<=match_tol_&&s.confidence>=min_conf_;}
  void step(){SurfaceSample s;if(!select(s)){out_->send(fx_.update(),fz_.update(),fr_.update());return;}double dt=1/sensor_rate_;
    Vec3 fb{px_.update(cx_.sp,s.measured.x,dt),pz_.update(cz_.sp,s.measured.z,dt),pr_.update(cr_.sp,s.measured.ry,dt)};
    Vec3 u{cx_.ff_gain*s.feedforward.x+fb.x,cz_.ff_gain*s.feedforward.z+fb.z,cr_.ff_gain*s.feedforward.ry+fb.ry};
    fx_.setTarget(std::clamp(u.x,cx_.lo,cx_.hi));fz_.setTarget(std::clamp(u.z,cz_.lo,cz_.hi));fr_.setTarget(std::clamp(u.ry,cr_.lo,cr_.hi));out_->send(fx_.update(),fz_.update(),fr_.update());}
  json j_;AxisCfg cx_,cz_,cr_;PID px_,pz_,pr_;FOH fx_,fz_,fr_;std::deque<SurfaceSample>buf_;size_t max_samples_{2000};double match_tol_{.005},max_age_{10},min_conf_{.5},delay_{0},rate_,sensor_rate_,tcp_x_{0},tcp_y_{0},tcp_z_{0};bool have_tcp_{false};std::string mode_;std::unique_ptr<FanucDpmServer>out_;rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr surf_,tcp_;rclcpp::TimerBase::SharedPtr timer_;};
int main(int argc,char**argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<Node>());rclcpp::shutdown();}
