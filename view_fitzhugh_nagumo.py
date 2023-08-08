import numpy as np
import matplotlib.pyplot as plt
from libs import runge_kutta_step, fitzhugh_nagumo

T = 2000
dt = 0.1
pulse_strength = 0.3
pulse_interval = [10, 100, 300]

for interval in pulse_interval:

    x = [0., 0.]
    x = np.array(x)

    ts = np.arange(T) * dt
    pulse_input = pulse_strength * ((np.arange(T) // interval) % 2) 
    record = np.zeros((T, 2))

    for idx, t in enumerate(ts):
        record[idx] = x
        x = runge_kutta_step(
            fitzhugh_nagumo, x, t, dt,
            a=0.7, b=0.8, c=10, Iext=pulse_input[idx])

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                           gridspec_kw={'height_ratios': [3, 1]})
    ax[0].plot(ts, record[:, 0], label='u (membrane potential)')
    ax[0].plot(ts, record[:, 1], label='v (refractory dynamics)')
    ax[1].plot(ts, pulse_input, label='input signal')
    ax[1].set_xlabel('Time')
    ax[0].legend()
    ax[1].legend()
    plt.savefig(f"figs/view_fitzhugh_nagumo_{interval}.png")
