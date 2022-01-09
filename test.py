from CSTR import CSTRsys
import matplotlib.pyplot as plt
import numpy as np


Sys=CSTRsys()

Sys.initial_conditions()

Sys.sym_step(tf=10000,t0=0,up=np.array([[200,0],[500,1],[1500,0.5],[2500,0.9],[3500,1]]),dT=0.1)

plt.plot(Sys.t_data,Sys.z_data[3,:],'r')
plt.plot(Sys.t_data,Sys.z_data[2,:],'g')
plt.plot(Sys.t_data,Sys.z_data[1,:],'g')
plt.plot(Sys.t_data,Sys.z_data[0,:],'g')
plt.plot(Sys.t_data,Sys.u_data,'b')
#plt.plot(Sys.u_data)
#plt.ylim(-0.1,1.2)
plt.show()
