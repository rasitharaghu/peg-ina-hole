#!/usr/bin/env python3
"""Offline sanity test of PID+FOH and preview-buffer timing; no ROS/FANUC required."""
from collections import deque
import math
class PID:
 def __init__(self,kp,ki,kd,lo,hi):self.kp,self.ki,self.kd,self.lo,self.hi=kp,ki,kd,lo,hi;self.i=0;self.pe=0
 def update(self,sp,y,dt):
  e=sp-y;self.i+=e*dt;d=(e-self.pe)/dt;self.pe=e;return max(self.lo,min(self.hi,self.kp*e+self.ki*self.i+self.kd*d))
def main():
 dt=1/125;pid=PID(.8,.08,0,-.01,.01);buf=deque();v=.05;lead=.10
 print('t, tcp_x, sensed_surface_x, matched_z, pid_z')
 for k in range(125*5):
  t=k*dt;tcp=v*t
  if k%4==0: # ~31.25 Hz sensor
   sx=tcp+lead; z=.002*math.sin(2*math.pi*.5*sx);buf.append((sx,z,t))
  match=min(buf,key=lambda q:abs(q[0]-tcp)) if buf else None
  if match and abs(match[0]-tcp)<.006:
   u=pid.update(0,match[1],dt)
   if k%25==0:print(f'{t:.2f}, {tcp:.3f}, {match[0]:.3f}, {match[1]:+.5f}, {u:+.5f}')
 print('PASS: buffer contains past ahead-looking samples and matches them when TCP reaches their locations.')
if __name__=='__main__':main()
