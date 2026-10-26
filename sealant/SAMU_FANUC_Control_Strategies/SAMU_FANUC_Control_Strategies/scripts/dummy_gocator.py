#!/usr/bin/env python3
"""Dummy GoCator for all three strategies. SI units: m, rad."""
import math, rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
class Dummy(Node):
 def __init__(self):
  super().__init__('dummy_gocator');self.t=0.0
  self.fb=self.create_publisher(Float64MultiArray,'/samu/gocator/feedback_x_z_ry',10)
  self.ff=self.create_publisher(Float64MultiArray,'/samu/gocator/feedforward_x_z_ry',10)
  self.surf=self.create_publisher(Float64MultiArray,'/samu/gocator/surface_sample',10)
  self.timer=self.create_timer(1/30,self.tick)
 def pub(self,p,a):m=Float64MultiArray();m.data=list(map(float,a));p.publish(m)
 def tick(self):
  self.t+=1/30
  # synthetic residual measurement errors
  x=0.001*math.sin(2*math.pi*.15*self.t); z=0.002*math.sin(2*math.pi*.10*self.t); ry=math.radians(2)*math.sin(2*math.pi*.08*self.t)
  # synthetic geometry-derived feedforward correction
  ff=[-0.5*x,-0.8*z,-0.7*ry]
  self.pub(self.fb,[x,z,ry]);self.pub(self.ff,ff)
  # sample location in common/workpiece frame, intentionally ahead of dummy dispenser by 0.10 m in X
  px=0.05*self.t+0.10; self.pub(self.surf,[px,0.0,0.0,x,z,ry,*ff,1.0])
def main():rclpy.init();n=Dummy();rclpy.spin(n);n.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
