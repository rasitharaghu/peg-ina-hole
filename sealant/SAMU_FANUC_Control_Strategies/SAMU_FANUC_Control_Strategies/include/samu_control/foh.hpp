#pragma once
#include <algorithm>
#include <cmath>
class FOH {
 public:
  void configure(double dt,double source_period,double max_rate,double max_accel){
    dt_=dt; period_=source_period; vmax_=std::abs(max_rate); amax_=std::abs(max_accel); reset();
  }
  void reset(double x=0){x_=target_=start_=x;v_=0;elapsed_=0;}
  void setTarget(double t){start_=x_; target_=t; elapsed_=0;}
  double update(){
    elapsed_+=dt_;
    double vdes;
    if(period_>0 && elapsed_<period_) vdes=(target_-start_)/period_;
    else vdes=(target_-x_)/std::max(dt_,1e-9);
    vdes=std::clamp(vdes,-vmax_,vmax_);
    const double dv=std::clamp(vdes-v_,-amax_*dt_,amax_*dt_);
    v_+=dv; x_+=v_*dt_;
    if(std::abs(target_-x_)<1e-12){x_=target_;v_=0;}
    return x_;
  }
 private:
  double dt_{.008},period_{.033},vmax_{1},amax_{10},x_{0},target_{0},start_{0},v_{0},elapsed_{0};
};
