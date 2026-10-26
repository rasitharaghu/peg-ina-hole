#pragma once
#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
class FanucDpmServer {
 public:
  FanucDpmServer(bool dry,std::string host,int port,bool ry_to_p,double lin_scale,double ang_scale)
   :dry_(dry),host_(std::move(host)),port_(port),ry_to_p_(ry_to_p),ls_(lin_scale),as_(ang_scale){if(!dry_) start();}
  ~FanucDpmServer(){if(client_>=0)close(client_);if(server_>=0)close(server_);}
  void send(double dx,double dz,double dry){
    // FANUC DPM order used by supplied KAREL example: dx,dy,dz,dw,dp,dr.
    // ry_to_p is CONFIGURABLE because process Ry -> FANUC P is frame dependent and must be verified.
    std::array<float,6> v{float(dx*ls_),0.0f,float(dz*ls_),0.0f,0.0f,0.0f};
    if(ry_to_p_)v[4]=float(dry*as_); else v[3]=float(dry*as_);
    if(dry_){last_=v;return;}
    poll_client(); if(client_<0||!triggered_)return;
    ::send(client_,v.data(),sizeof(v),MSG_NOSIGNAL); last_=v;
  }
  const std::array<float,6>& last()const{return last_;}
 private:
  void start(){server_=socket(AF_INET,SOCK_STREAM,0);int one=1;setsockopt(server_,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
    sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(port_);a.sin_addr.s_addr=inet_addr(host_.c_str());
    if(bind(server_,(sockaddr*)&a,sizeof(a))<0)throw std::runtime_error("DPM bind failed");listen(server_,1);fcntl(server_,F_SETFL,O_NONBLOCK);}
  void poll_client(){if(client_<0){client_=accept(server_,nullptr,nullptr);if(client_>=0)fcntl(client_,F_SETFL,O_NONBLOCK);}
    if(client_>=0&&!triggered_){uint32_t t=0;auto n=recv(client_,&t,4,MSG_DONTWAIT);if(n==4&&t==42)triggered_=true;}}
  bool dry_{true},ry_to_p_{true},triggered_{false};std::string host_;int port_{30000},server_{-1},client_{-1};double ls_{1000},as_{57.295779513};std::array<float,6>last_{};
};
