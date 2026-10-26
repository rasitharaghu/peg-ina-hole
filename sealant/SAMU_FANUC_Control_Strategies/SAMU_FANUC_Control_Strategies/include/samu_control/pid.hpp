#pragma once
#include <algorithm>
class PID {
 public:
  PID()=default;
  PID(double kp,double ki,double kd,double lo,double hi,double n=10.0)
      :kp_(kp),ki_(ki),kd_(kd),lo_(lo),hi_(hi),n_(n){}
  void configure(double kp,double ki,double kd,double lo,double hi,double n=10.0){
    kp_=kp;ki_=ki;kd_=kd;lo_=lo;hi_=hi;n_=n;reset();
  }
  void reset(){i_=0;prev_e_=0;prev_d_=0;first_=true;}
  double update(double sp,double y,double dt){
    if(dt<=0) return last_;
    const double e=sp-y;
    const double p=kp_*e;
    i_ += e*dt;
    const double raw=first_?0.0:(e-prev_e_)/dt;
    const double a=dt/(dt+1.0/std::max(n_,1e-9));
    const double df=a*raw+(1-a)*prev_d_;
    double u=p+ki_*i_+kd_*df;
    const double sat=std::clamp(u,lo_,hi_);
    if(sat!=u && ((u>hi_&&e>0)||(u<lo_&&e<0))) i_-=e*dt;
    prev_e_=e; prev_d_=df; first_=false; last_=sat; return sat;
  }
 private:
  double kp_{0},ki_{0},kd_{0},lo_{-1e9},hi_{1e9},n_{10};
  double i_{0},prev_e_{0},prev_d_{0},last_{0}; bool first_{true};
};
