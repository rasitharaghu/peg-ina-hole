#pragma once
#include <array>
#include <chrono>
struct Vec3 { double x{0}, z{0}, ry{0}; };
struct SurfaceSample {
  double stamp{0};
  // Location of the surface sample in the configured common/workpiece frame.
  double px{0}, py{0}, pz{0};
  // Measured/estimated process values at that location.
  Vec3 measured{};
  Vec3 feedforward{};
  double confidence{1.0};
};
inline Vec3 operator+(const Vec3&a,const Vec3&b){return {a.x+b.x,a.z+b.z,a.ry+b.ry};}
inline Vec3 operator*(const Vec3&a,double k){return {a.x*k,a.z*k,a.ry*k};}
