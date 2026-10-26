#!/usr/bin/env python3
"""Dummy dispenser TCP in same common/workpiece frame as surface_sample."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
class DummyTCP(Node):
 def __init__(self):super().__init__('dummy_tcp');self.t=0.;self.p=self.create_publisher(Float64MultiArray,'/samu/dispenser/tcp_pose',10);self.timer=self.create_timer(1/125,self.tick)
 def tick(self):self.t+=1/125;m=Float64MultiArray();m.data=[0.05*self.t,0.,0.,0.];self.p.publish(m)
def main():rclpy.init();n=DummyTCP();rclpy.spin(n);n.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
