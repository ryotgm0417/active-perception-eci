import numpy as np
import matplotlib.pyplot as plt
from libs import runge_kutta_step, fitzhugh_nagumo

T = 1000
dt = 0.01
Iext = 0.5

x = [0., 0.]
x = np.array(x)

ts = np.arange(T) * dt
record = np.zeros((T, 2))

for idx, t in enumerate(ts):
    record[idx] = x
    x = runge_kutta_step(
        fitzhugh_nagumo, x, t, dt,
        a=0.7, b=0.8, c=10, Iext=Iext)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(ts, record[:, 0], label='u (membrane potential)')
ax.plot(ts, record[:, 1], label='v (refractory dynamics)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.savefig("figs/view_fitzhugh_nagumo.png")
