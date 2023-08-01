import numpy as np
import matplotlib.pyplot as plt
from libs import *

T = 100000
dt = 0.01

ts = np.arange(T) * dt
rec_Iin = np.zeros((T, 10))
rec_in = np.zeros((T, 10))
rec_hidden = np.zeros((T, 16))
rec_out = np.zeros((T, 4))
rec_motor = np.zeros((T, 4))
rec_agent_state = np.zeros((T, 3))

agent = EmbodiedAgent(grid_width=5, T=T, dt=dt)

for idx, t in enumerate(ts):
    # record data (1)
    rec_in[idx] = agent.net.Xin[:, 0]
    rec_hidden[idx] = agent.net.Xhidden[:, 0]
    rec_out[idx] = agent.net.Xout[:, 0]
    rec_agent_state[idx] = agent.X

    # advance simulation step
    _, data = agent.step()

    # record data (2)
    rec_Iin[idx] = data["Iin"]
    rec_motor[idx] = data["Motor"]

# plot
plot_range = slice(50000, 60000)

fig, ax = plt.subplots(figsize=(9, 9))
ax.plot(rec_agent_state[:, 0], rec_agent_state[:, 1],
        c='k', lw=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.savefig("figs/view_embodied_agent_pos.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts[plot_range], rec_Iin[plot_range], c='m', lw=1, label='input signal')
ax.plot(ts[plot_range], rec_in[plot_range], c='r', lw=1, label='input neurons')
ax.plot(ts[plot_range], rec_hidden[plot_range], c='g', lw=1, label='hidden neurons')
ax.plot(ts[plot_range], rec_out[plot_range], c='b', lw=1, label='output neurons')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
handles, labels = ax.get_legend_handles_labels()
display = (0, 10, 20, 36)
ax.legend([handle for i, handle in enumerate(handles) if i in display],
          [label for j, label in enumerate(labels) if j in display])
plt.savefig("figs/view_embodied_agent_nn.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts[plot_range], rec_motor[plot_range], c='b', lw=1.5, label='motor output')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[0]], labels[0])
plt.savefig("figs/view_embodied_agent_motor.png")
