import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from libs import EmbodiedAgent, visualize

T = 10000
dt = 0.1
grid_width = 50

ts = np.arange(T) * dt
rec_Iin = np.zeros((T, 10))
rec_in = np.zeros((T, 10))
rec_hidden = np.zeros((T, 16))
rec_out = np.zeros((T, 4))
rec_motor = np.zeros((T, 4))
rec_agent_state = np.zeros((T, 3))
rec_sensor_pos = np.zeros((T, 10, 2))
rec_sensor_activation = np.zeros((T, 10))

agent = EmbodiedAgent(grid_width=grid_width, T=T, dt=dt)

for idx, t in enumerate(tqdm(ts)):
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
    rec_sensor_pos[idx] = data["Sensor_pos"]
    rec_sensor_activation[idx] = data["Sensor_activation"]

# plot
plot_range = slice(0, 200)

fig, ax = plt.subplots(figsize=(9, 9))
ax.plot(rec_agent_state[:, 0], rec_agent_state[:, 1],
        c='k', lw=2)
ax.set_aspect('equal')
xmin, xmax = ax.get_xlim()
xmin = int(xmin // grid_width) + 1
xmax = int(xmax // grid_width) + 1
ymin, ymax = ax.get_ylim()
ymin = int(ymin // grid_width) + 1
ymax = int(ymax // grid_width) + 1
for i in range(xmin, xmax):
    ax.axvline(i*grid_width, lw=1)
for i in range(ymin, ymax):
    ax.axhline(i*grid_width, lw=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.savefig("figs/view_embodied_agent_pos.png")

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6),
                       gridspec_kw={'height_ratios': [3, 1]})
ax[0].plot(ts[plot_range], rec_in[plot_range], c='r', lw=1, label='input neurons')
ax[0].plot(ts[plot_range], rec_hidden[plot_range], c='g', lw=1, label='hidden neurons')
ax[0].plot(ts[plot_range], rec_out[plot_range], c='b', lw=1, label='output neurons')
ax[0].set_ylabel('Membrane Potential')
ax[1].plot(ts[plot_range], rec_Iin[plot_range], c='m', lw=1, label='input signal')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Value')
handles0, labels0 = ax[0].get_legend_handles_labels()
handles1, labels1 = ax[1].get_legend_handles_labels()
display = (0, 10, 26)
ax[0].legend([handle for i, handle in enumerate(handles0) if i in display],
          [label for j, label in enumerate(labels0) if j in display])
ax[1].legend([handles1[0]], [labels1[0]])
plt.savefig("figs/view_embodied_agent_nn.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts[plot_range], rec_motor[plot_range], c='b', lw=1.5, label='motor output')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[0]], [labels[0]])
plt.savefig("figs/view_embodied_agent_motor.png")

visualize(rec_agent_state, rec_sensor_pos, rec_sensor_activation, grid_width,
          file_path="figs/animation.mp4")
