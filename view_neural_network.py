import numpy as np
import matplotlib.pyplot as plt
from libs import FitzhughNagumoNetwork

T = 10000
plot_length = 2000
dt = 0.01
np.random.seed(1)

ts = np.arange(T) * dt
arr_Iin = 0.28 + 0.1*np.random.rand(T, 10)
rec_in = np.zeros((T, 10))
rec_hidden = np.zeros((T, 16))
rec_out = np.zeros((T, 4))
rec_motor = np.zeros((T, 4))

net = FitzhughNagumoNetwork(
    T=T, dt=dt, Nin=10, Nhidden=16, Nout=4,
    delay_in=10, delay_out=10, pulse_max=0.7,
    pulse_min=0.0, motor_amp=1.5, pulse_width=20,
    p=0.2, seed=0
)

for idx, t in enumerate(ts):
    Iin = arr_Iin[idx]
    net.propagate_pulse()
    Ihidden, Iout, Motor = net.compute_current_signal()

    # record data
    rec_in[idx] = net.Xin[:, 0]
    rec_hidden[idx] = net.Xhidden[:, 0]
    rec_out[idx] = net.Xout[:, 0]
    rec_motor[idx] = Motor

    # advance simulation
    net.step(Iin, Ihidden, Iout)

# plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts[:plot_length], arr_Iin[:plot_length], c='m', lw=1, label='input signal')
ax.plot(ts[:plot_length], rec_in[:plot_length], c='r', lw=1, label='input neurons')
ax.plot(ts[:plot_length], rec_hidden[:plot_length], c='g', lw=1, label='hidden neurons')
ax.plot(ts[:plot_length], rec_out[:plot_length], c='b', lw=1, label='output neurons')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
handles, labels = ax.get_legend_handles_labels()
display = (0, 10, 20, 36)
ax.legend([handle for i, handle in enumerate(handles) if i in display],
          [label for j, label in enumerate(labels) if j in display])
plt.savefig("figs/view_neural_network_nn.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts[:plot_length], rec_motor[:plot_length], c='b', lw=1.5, label='motor output')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[0]], labels[0])
plt.savefig("figs/view_neural_network_motor.png")
