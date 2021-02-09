# code adapted from https://github.com/mackelab/delfi/blob/main/delfi/simulator/TwoMoons.py
import numpy as np
import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
torch.pi_val = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
torch.pi_tensor = torch.acos(torch.zeros(1)) * 2  # which is 3.1415927410125732


def default_mapfunc(theta, p):
    ang = -np.pi / 4.0
    c = np.cos(ang)
    s = np.sin(ang)  # this is the scaling
    z0 = c * theta[0] - s * theta[1]
    z1 = s * theta[0] + c * theta[1]
    return p + np.array([-np.abs(z0), z1])


def default_mapfunc_inverse(theta, x):
    # ang = -np.pi / 4.0
    # c = np.cos(ang)
    # s = np.sin(ang)
    # z0 = c * theta[0] - s * theta[1]
    # z1 = s * theta[0] + c * theta[1]
    ang = -torch.pi_tensor / 4
    c = torch.cos(ang)
    s = torch.sin(ang)
    z0 = c * theta[0] - s * theta[1]
    z1 = s * theta[0] + c * theta[1]

    return x - torch.cat((-torch.abs(z0), z1), 0)


class TwoMoons():

    def __init__(self, prior, mean_radius=1.0, sd_radius=0.1, baseoffset=1.0, fixed_data=True):

        self.prior = prior
        self.mean_radius = mean_radius
        self.sd_radius = sd_radius
        self.baseoffset = baseoffset
        self.fixed_data = fixed_data  # if fixe_data = True, then the data is fixed to x_o = [0,0], ow data is simulated from the forward model

        self.mapfunc = default_mapfunc
        self.mapfunc_inverse = default_mapfunc_inverse
        # self.mapfunc_Jacobian_determinant = default_mapfunc_Jacobian_determinant

    def likelihood(self, param, x, log=True):
        # assert x.size == 2, "not yet implemented for evaluation on multiple points at once"
        # assert np.isfinite(x).all() and (np.imag((x)) == 0).all(), "invalid input"
        # if self.mapfunc_inverse is None or self.mapfunc_Jacobian_determinant is None:
        #    return np.nan
        # param = param.detach().numpy()
        # x = x.detach().numpy()

        p = default_mapfunc_inverse(param, x)
        # assert p.size == 2, "not yet implemented for non-bijective map functions"
        u = p[0] - self.baseoffset
        v = p[1]

        # if u < 0.0:  # invalid x for this theta
        #    return -np.inf if log else 0.0

        if u < 0.0:  # invalid x for this theta
            return -torch.Tensor([float("Inf")]) if log else torch.Tensor([0.0])

        # r = np.sqrt(u ** 2 + v ** 2)  # note the angle distribution is uniform
        r = torch.sqrt(u.pow(2) + v.pow(2))
        # L = -0.5 * ((r - self.mean_radius) / self.sd_radius) ** 2 - 0.5 * np.log(2 * np.pi * self.sd_radius ** 2)
        L = -0.5 * ((r - self.mean_radius) / self.sd_radius).pow(2) - 0.5 * torch.log(2 * torch.pi_tensor *
                                                                                      self.sd_radius ** 2)

        return L if log else torch.exp(L)

    def gen_single(self, param):

        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        # assert param.ndim == 1
        # assert param.shape[0] == self.dim_param

        # a = np.pi * (self.rng.rand() - 0.5)
        # r = self.mean_radius + self.rng.randn() * self.sd_radius

        a = np.pi * (np.random.uniform() - 0.5)
        r = self.mean_radius + np.random.normal() * self.sd_radius
        p = np.array([r * np.cos(a) + self.baseoffset, r * np.sin(a)])
        return self.mapfunc(param, p)

    def gen_posterior_samples(self, obs,
                              n_samples=1):  # rm prior (since prior is uniform??) , prior=None, n_samples=1):
        # works only when we use the default_mapfunc above

        # TODO: what exactly is happening here??

        obs = obs.numpy()

        # use opposite rotation as above
        ang = -np.pi / 4.0
        c = np.cos(-ang)
        s = np.sin(-ang)

        theta = np.zeros((n_samples, 2))
        for i in range(n_samples):
            p = self.gen_single(np.zeros(2))  # ['data']
            q = np.zeros(2)
            q[0] = p[0] - obs[0]
            q[1] = obs[1] - p[1]

            if np.random.rand() < 0.5:
                q[0] = -q[0]

            theta[i, 0] = c * q[0] - s * q[1]
            theta[i, 1] = s * q[0] + c * q[1]

        return torch.from_numpy(theta).to(dtype=torch.float32)  # return torch tensor

    # function to sample form the prior pred dist
    def gen_prior_pred(self, n, dim=2):

        # theta_samples = self.prior.sample(n)

        theta_samples = self.prior.rsample(sample_shape=(n,))

        x_samples = torch.zeros(n, dim)

        for i in range(n):
            x_samples[i, :] = torch.from_numpy(self.gen_single(theta_samples[i, :])).to(dtype=torch.float32)

        return x_samples, theta_samples

    # function to gen data give theta values
    def model_sim(self, theta, dim=2):

        n = theta.shape[0]

        x_samples = torch.zeros(n, dim)

        for i in range(n):
            x_samples[i, :] = torch.from_numpy(self.gen_single(theta[i, :])).to(dtype=torch.float32)

        return x_samples
