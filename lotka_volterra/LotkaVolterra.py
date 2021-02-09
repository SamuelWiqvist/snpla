import torch
from tqdm import tqdm
from torch.distributions.exponential import Exponential
from torch.distributions.categorical import Categorical


# Code that implements the LV model from "Sequential Neural Likelihood" (SNL) (https://arxiv.org/abs/1805.07226).
# The code is inspired by the code provided in the sup. material for the paper "Fast ε-free Inference"
# (https://papers.nips.cc/paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html)


def gen_exponential(lam):
    """
    Draws one sample from Exp(lam) (lam is the rate)

    :param lam: rate for the exponential dist
    :return: one sample from Exp(lam)
    """
    return -torch.rand(1).log() / lam


def calc_summary_stats(paths):
    """
    Calcs the summary stats according to the SNL paper

    :param paths: list with paths
    :return: tensor of size nx9
    """

    # nbr_paths = paths.shape[0]
    # nbr_samples = paths.shape[-1]

    nbr_paths = len(paths)

    summary_stats = torch.zeros((nbr_paths, 9))

    for i in range(nbr_paths):
        nbr_samples = paths[i].shape[1]  # nbr samples for current path

        x_tmp = paths[i][0, :]
        y_tmp = paths[i][1, :]

        mean_x = x_tmp.mean()
        mean_y = y_tmp.mean()

        var_x = x_tmp.var()
        var_y = y_tmp.var()

        summary_stats[i, 0] = mean_x
        summary_stats[i, 4] = mean_y

        summary_stats[i, 1] = (var_x + 1).log()
        summary_stats[i, 5] = (var_y + 1).log()

        # demean instead of standardizing
        x_tmp = (x_tmp - mean_x)/torch.sqrt(var_x)
        y_tmp = (y_tmp - mean_y)/torch.sqrt(var_y)

        summary_stats[i, 2] = torch.dot(x_tmp[1:], x_tmp[:-1]) / (nbr_samples - 1)
        summary_stats[i, 3] = torch.dot(x_tmp[2:], x_tmp[:-2]) / (nbr_samples - 1)
        summary_stats[i, 6] = torch.dot(y_tmp[1:], y_tmp[:-1]) / (nbr_samples - 1)
        summary_stats[i, 7] = torch.dot(y_tmp[2:], y_tmp[:-2]) / (nbr_samples - 1)
        summary_stats[i, 8] = torch.dot(x_tmp, y_tmp) / (nbr_samples - 1)

        # 0 mean(x1),
        # 1 log(var(x1) + 1),
        # 2 autocorr_order2(x1),
        # 3 autocorr_order3(x1),
        # 4 mean(x2),
        # 5 log(var(x2) + 1),
        # 6 autocorr_order2(x2),
        # 7 autocorr_order3(x2)
        # 8 corr_between_x1, x2

    return summary_stats


def update_state(reac, x_current, y_current):
    if reac == 0:
        x_new = x_current + 1
        y_new = y_current
    elif reac == 1:
        x_new = x_current - 1
        y_new = y_current
    elif reac == 2:
        x_new = x_current
        y_new = y_current + 1
    else:
        x_new = x_current
        y_new = y_current - 1

    return x_new, y_new


class LotkaVolterra:
    def __init__(self, prior, dur=30, dt=0.2, max_steps=10000):
        self.prior = prior
        self.dur = dur
        self.dt = dt
        self.max_steps = max_steps

    def model_sim(self, theta, print_progress=False):
        """
        Runs model sims and returns the summary stats

        :param theta: tensor of size nx4 with param values
        :param print_progress: flag for printing progress (default False)
        :return: tensor of size nx9 with summary stats gen. at param val
        """
        # TODO shoulded the normalization be added here??
        return calc_summary_stats(self.model_sim_paths(theta, print_progress))

    def model_sim_paths(self, theta, print_progress=False):
        """
        Simulate from LV model for a given theta

        :param theta: tensor of size nx4 with param values
        :param print_progress: flag for printing progress (default False)
        :return: paths: tensor of size nx2x151 with paths gen. at param val
        """
        n = theta.shape[0]

        # paths = torch.zeros(n, 2, int(self.dur / self.dt) + 1)

        paths = []
        for i in tqdm(range(n), disable=not print_progress, desc="Running " + str(n) + " simulations"):
            while True:
                try:
                    # paths[i, :, :] = self.gen_single(theta[i, :])
                    paths.append(self.gen_single(theta[i, :]))
                    break
                except Exception as e:
                    print(e)
                    print("Redo sim")

        return paths

    def gen_single(self, theta, x_start=50, y_start=100):
        """
        Simulates on path for the LV model at the param values theta using the gillespie algorithm

        :param self:
        :param theta: param values torch(4) vector
        :param x_start: initial pop. (default 50)
        :param y_start: initial pop. (default 100)
        :return: path: path sim at theta
        """
        path = torch.zeros((2, int(self.dur / self.dt) + 1))
        nbr_steps = 0
        num_sim_steps = int(self.dur / self.dt)
        current_time = self.dt
        time = 0

        path[0, 0] = x_start
        path[1, 0] = y_start

        for i in range(num_sim_steps):

            x_current = path[0, i]
            y_current = path[1, i]

            while current_time > time:

                r1 = torch.exp(theta[0]) * x_current * y_current
                r2 = torch.exp(theta[1]) * x_current
                r3 = torch.exp(theta[2]) * y_current
                r4 = torch.exp(theta[3]) * x_current * y_current

                r_sum = r1 + r2 + r3 + r4

                # TODO: what to do if r_sum = 0?????
                if r_sum == 0:
                    time = float('inf')  # this will let the process cont but die out
                    break

                time = time + gen_exponential(r_sum)

                reac = torch.multinomial(torch.tensor([r1, r2, r3, r4]) / r_sum, 1)
                # this can prob be done more efficient see:
                # http://www.dcs.gla.ac.uk/~srogers/teaching/mscbioinfo/SysBio2.pdf

                x_current, y_current = update_state(reac, x_current, y_current)

                nbr_steps = nbr_steps + 1

                if nbr_steps > self.max_steps:
                    # TODO what should happen here??? # <- this step is prob not correct!
                    # TODO what to do with a case where this happens, just remove that samples or restart??
                    # TODO maybe we can run over nbr of steps instead?? But what then to do if the pop dies out??
                    # raise Exception("Simulation too long!")
                    print("Simulation too long!")
                    return path  # here we simply return the current pop with zeros for the rest of the time steps

            path[0, i + 1] = x_current
            path[1, i + 1] = y_current
            current_time = current_time + self.dt

        return path
