#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
using json=nlohmann::json;
inline json load_json(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error("Cannot open config: "+p);return json::parse(f);}
struct AxisCfg{double sp{0},kp{0},ki{0},kd{0},lo{-1},hi{1},rate{1},accel{10},ff_gain{1};};
inline AxisCfg axis_cfg(const json&j,const char*n){auto a=j.at("axes").at(n);return {
 a.value("setpoint",0.0),a.value("kp",0.0),a.value("ki",0.0),a.value("kd",0.0),
 a.value("min_offset",-1.0),a.value("max_offset",1.0),a.value("max_rate",1.0),
 a.value("max_accel",10.0),a.value("feedforward_gain",1.0)};}
