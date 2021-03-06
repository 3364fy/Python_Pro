import matplotlib.pyplot as plt
import numpy as np

x=np.arange(0,10,0.00001)
y1=(x-x**2)*(np.sin(x))**2

plt.plot(x,y1,label='120')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()