#pragma once
#include <std_msgs/msg/float64_multi_array.hpp>
#include "samu_control/types.hpp"
inline Vec3 vec3msg(const std_msgs::msg::Float64MultiArray&m){Vec3 v; if(m.data.size()>0)v.x=m.data[0];if(m.data.size()>1)v.z=m.data[1];if(m.data.size()>2)v.ry=m.data[2];return v;}
