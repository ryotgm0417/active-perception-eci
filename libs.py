import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def runge_kutta_step(func, x, t, dt, **params):
    '''Compute 1 step of 4th-order
    Runge-Kutta method (RK4)

    Args:
        func (function): system equation
        (dx/dt = func(x, t, **params))
        x (np.ndarray or list): system state
        t (float): current time
        dt (float): timestep
    '''
    x = np.array(x)
    k1 = dt * func(x, t, **params)
    k2 = dt * func(x + 0.5 * k1, t + 0.5 * dt, **params)
    k3 = dt * func(x + 0.5 * k2, t + 0.5 * dt, **params)
    k4 = dt * func(x + k3, t + dt, **params)
    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


def fitzhugh_nagumo(x, t, Iext=0.7, a=0.7, b=0.8, c=10):
    '''Fitzhugh nagumo neuron model

    Args:
        x: state (membrane potential, refractory dynamics)
        t: time
        Iext: external signal
        a, b, c: system parameters
    '''
    x_dot = np.zeros_like(x)
    x_dot[0] = c * (x[0] - x[0]**3/3 - x[1] + Iext)
    x_dot[1] = a + x[0] - b*x[1]
    return x_dot


def navigation_dynamics(x, t, FL=0, FR=0, g1=5, g2=50):
    '''Equation for the agent navigation motion

    Args:
        x: state (x and y position, heading direction)
        t: time
        FL, FR: forward forces
        g1, g2: system parameters
    '''
    x_dot = np.zeros_like(x)
    x_dot[0] = g2 * (FL + FR) * np.cos(x[2])
    x_dot[1] = g2 * (FL + FR) * np.sin(x[2])
    x_dot[2] = g1 * (FL - FR)
    return x_dot


class FitzhughNagumoNetwork(object):
    '''Neural network composed of Fitzhugh-Nagumo neurons

    Attributes:
        T (int): total number of time steps to simulate
        dt (float): simulation time step
        Nin (int): number of input neurons
        Nhidden (int): number of hidden (internal) neurons
        Nout (int): number of output neurons
        delay (int): delay of neuron signal propagation
        p (float): connecting probability between neurons
        seed (int): random seed
    '''
    def __init__(self, T=10000, dt=0.01,
                 Nin=10, Nhidden=16, Nout=4,
                 delay=10, pulse_max=0.7, pulse_min=0.0,
                 motor_amp=1.5, p=0.2, seed=0):

        self.T = T
        self.dt = dt
        self.Nin = Nin
        self.Nhidden = Nhidden
        self.Nout = Nout
        self.delay = delay
        self.pulse_max = pulse_max
        self.pulse_min = pulse_min
        self.motor_amp = motor_amp
        self.p = p
        self.rnd = np.random.RandomState(seed)

        # initialize state of each neuron
        # self.Xin = np.zeros((Nin, 2))
        # self.Xhidden = np.zeros((Nhidden, 2))
        # self.Xout = np.zeros((Nout, 2))
        self.Xin = self.rnd.uniform(size=(Nin, 2))
        self.Xhidden = self.rnd.uniform(size=(Nhidden, 2))
        self.Xout = self.rnd.uniform(size=(Nout, 2))
        self.steps = 0

        # define connectivity matrix
        self.W_in_hidden = self.rnd.uniform(0, 1, (Nhidden, Nin)) < p
        self.W_hidden_out = self.rnd.uniform(0, 1, (Nout, Nhidden)) < p
        self.W_hidden_hidden = self.rnd.uniform(0, 1, (Nhidden, Nhidden)) < p
        self.W_out_hidden = self.rnd.uniform(0, 1, (Nhidden, Nout)) < p

        # pulse signals
        self.pulse_hidden = np.zeros((T, Nhidden))
        self.pulse_out = np.zeros((T, Nout))
        self.pulse_motor = np.zeros((T, Nout))

    def step(self, Iin, Ihidden, Iout):
        t = self.steps * self.dt
        for i in range(self.Nin):
            self.Xin[i] = runge_kutta_step(
                fitzhugh_nagumo, self.Xin[i], t, self.dt,
                Iext=Iin[i])
        for i in range(self.Nhidden):
            self.Xhidden[i] = runge_kutta_step(
                fitzhugh_nagumo, self.Xhidden[i], t, self.dt,
                Iext=Ihidden[i])
        for i in range(self.Nout):
            self.Xout[i] = runge_kutta_step(
                fitzhugh_nagumo, self.Xout[i], t, self.dt,
                Iext=Iout[i])
        self.steps += 1
        return self.Xin, self.Xhidden, self.Xout

    def propagate_pulse(self):
        delayed_steps = min(self.steps + self.delay, self.T-1)

        # in -> hidden
        idx_hidden = np.dot(self.W_in_hidden, self.Xin[:, 0] > 0) > 0
        self.pulse_hidden[delayed_steps, idx_hidden] = 1

        # hidden -> out
        idx_out = np.dot(self.W_hidden_out, self.Xhidden[:, 0] > 0) > 0
        self.pulse_out[delayed_steps, idx_out] = 1

        # hidden -> hidden
        idx_hidden = np.dot(self.W_hidden_hidden, self.Xhidden[:, 0] > 0) > 0
        self.pulse_hidden[delayed_steps, idx_hidden] = 1

        # out -> hidden
        idx_hidden = np.dot(self.W_out_hidden, self.Xout[:, 0] > 0) > 0
        self.pulse_hidden[delayed_steps, idx_hidden] = 1

        # out -> motor
        idx_motor = self.Xout[:, 0] > 0
        self.pulse_motor[self.steps, idx_motor] = 1

    def compute_current_signal(self):
        Ihidden = self.pulse_hidden[self.steps] * self.pulse_max + \
            (1 - self.pulse_hidden[self.steps]) * self.pulse_min
        Iout = self.pulse_out[self.steps] * self.pulse_max + \
            (1 - self.pulse_out[self.steps]) * self.pulse_min
        Motor = self.pulse_motor[self.steps] * self.motor_amp
        return Ihidden, Iout, Motor


class EmbodiedAgent(object):
    def __init__(self, radius=10,
                 sensor_max=0.28, sensor_min=0.21,
                 grid_width=20, **params):
        self.net = FitzhughNagumoNetwork(**params)

        # initial state of agent
        # X[0]: x position
        # X[1]: y position
        # X[2]: heading direction (theta)
        self.X = np.array([0., 0., 1.])
        self.steps = 0

        # simulation parameters
        self.T = self.net.T
        self.dt = self.net.dt

        # agent parameters
        self.radius = radius

        # environment parameters
        self.sensor_max = sensor_max
        self.sensor_min = sensor_min
        self.grid_width = grid_width

    def compute_sensor(self):
        vals = np.array([self.sensor_max, self.sensor_min])

        sensor_theta = np.arange(10) * np.pi / 5
        sensor_pos = np.zeros((10, 2))
        sensor_pos[:, 0] = self.X[0] + \
            self.radius * np.cos(self.X[2] + sensor_theta)
        sensor_pos[:, 1] = self.X[1] + \
            self.radius * np.sin(self.X[2] + sensor_theta)

        grid = np.mod(sensor_pos, float(2*self.grid_width))
        grid = (grid > self.grid_width)
        grid = np.sum(grid, axis=-1) % 2   # digital value

        Iin = vals[grid]
        return Iin, sensor_pos, grid

    def step(self):
        t = self.steps * self.dt
        Iin, sensor_pos, grid = self.compute_sensor()
        self.net.propagate_pulse()
        Ihidden, Iout, Motor = self.net.compute_current_signal()
        self.net.step(Iin, Ihidden, Iout)

        FL = np.tanh(Motor[0] + Motor[1])
        FR = np.tanh(Motor[2] + Motor[3])
        self.X = runge_kutta_step(
            navigation_dynamics, self.X, t, self.dt,
            FL=FL, FR=FR)
        self.steps += 1

        intermediate_data = {
            "Iin": Iin,
            "Ihidden": Ihidden,
            "Iout": Iout,
            "Motor": Motor,
            "Force": np.array([FL, FR]),
            "Sensor_pos": sensor_pos,
            "Sensor_activation": grid
        }
        return self.X, intermediate_data


def visualize(rec_agent_state, rec_sensor_pos, rec_sensor_activation, grid_width,
              stepsize=1, total_steps=2001, file_path="figs/animation.mp4"):
    
    print("Drawing animation...")
    fig, ax = plt.subplots(figsize=(9, 9))

    # Animation objects
    line, = ax.plot(rec_agent_state[:total_steps, 0],
                    rec_agent_state[:total_steps, 1],
                    c='k', lw=3)
    text = ax.text(0.1, 0.9, 'step 0',
                   fontsize='large',
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
    sensor_color = ['magenta' if rec_sensor_activation[0, i] == 0 else 'lime' \
                    for i in range(10)]
    coll = ax.scatter(rec_sensor_pos[0, :, 0], rec_sensor_pos[0, :, 1],
                       c=sensor_color, s=20)
    ax.set_aspect('equal')

    # Draw grid
    xmin, xmax = ax.get_xlim()
    xmin = int(xmin // grid_width) + 1
    xmax = int(xmax // grid_width) + 1
    ymin, ymax = ax.get_ylim()
    ymin = int(ymin // grid_width) + 1
    ymax = int(ymax // grid_width) + 1
    for i in range(xmin, xmax):
        ax.axvline(i*grid_width)
    for i in range(ymin, ymax):
        ax.axhline(i*grid_width)

    num_frames = min(total_steps, rec_agent_state.shape[0])
    num_frames = num_frames // stepsize

    def _animate(i):
        _end = stepsize*i + 1
        line.set_xdata(rec_agent_state[:_end, 0])
        line.set_ydata(rec_agent_state[:_end, 1])
        text.set_text(f'step {stepsize*i}')
        coll.set_offsets(np.vstack([rec_sensor_pos[_end-1, :, 0],
                                    rec_sensor_pos[_end-1, :, 1]]).T)
        sensor_color = ['magenta' if rec_sensor_activation[_end-1, i] == 0 \
                        else 'lime' for i in range(10)]
        coll.set_color(sensor_color)
        return line, text, coll

    ani = animation.FuncAnimation(
        fig, _animate, blit=True,
        frames=range(num_frames), interval=stepsize
    )

    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(file_path, writer=writer)
