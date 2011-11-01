import matplotlib
import numpy as np
import matplotlib.pyplot as plt
array_data=np.load('array_data.npz')

temps=array_data['temp']
press=array_data['press']
height=array_data['height']
avg_temp=np.mean(np.mean(temps,axis=1),axis=1)

fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(111)
ax.plot(avg_temp,height)
ax.set_xlabel('mean temperature (K)')
ax.set_ylabel('height (m)')
fig.savefig('figures/profile.png',dpi=150)
