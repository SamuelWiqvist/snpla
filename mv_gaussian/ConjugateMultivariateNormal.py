import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from .low_dim_w_summary_stats import functions as func


class ConjugateMultivariateNormal:
    def __init__(self, model, prior, N, x_dim):
        self.model = model
        self.prior = prior
        self.N = N
        self.dim = len(model.loc)
        self.x_dim = x_dim

    def sample(self, N=None):
        if N is None:
            N = self.N
        return self.model.rsample(sample_shape=(N,))

    def sample_fixed_seed(self, seed, N=None):
        torch.manual_seed(seed)
        if N is None:
            N = self.N
        return self.model.rsample(sample_shape=(N,))

    def model_sim(self, theta):
        """
        Simulate from model for a given theta

        :param theta: Matrix of size n x dim_theta
        :return: x_samples: Matrix of size x x dim_x with samples from the model
        """
        n = theta.shape[0]

        x_samples = torch.zeros(n, self.x_dim)

        for j in range(theta.shape[0]):
            model_tmp = MultivariateNormal(theta[j, :], self.model.covariance_matrix)
            x_samples[j, :] = model_tmp.rsample(sample_shape=(self.N,)).flatten()  # flatten here so I do not need to
            # flatten later

        return x_samples

    def calc_analytical_posterior(self, x_o):

        Simga_inv_0 = self.prior.precision_matrix
        Simga_inv = self.model.precision_matrix
        mu_0 = self.prior.loc
        x_bar = torch.transpose(x_o.mean(0, True), 0, 1)
        n = self.N

        mu_post = torch.mm((Simga_inv_0 + n * Simga_inv).inverse(),
                           torch.mm(n * Simga_inv, x_bar) + torch.mm(Simga_inv_0, mu_0.resize(len(mu_0), 1)))
        Sigma_inv_post = Simga_inv_0 + n * Simga_inv

        return MultivariateNormal(mu_post.resize(len(mu_post)), precision_matrix=Sigma_inv_post)

    def sample_analytical_posterior(self, analytical_posterior, N):
        return analytical_posterior.rsample(sample_shape=(N,))

    def kl_div(self, analytical_posterior, post_samples):
        # post_samples = posterior_samples_iter[1]

        d = post_samples.size()[1]

        mu_analytical = analytical_posterior.loc.numpy()
        prec_analytical = analytical_posterior.precision_matrix.numpy()

        mu_post = post_samples.mean(0).detach().numpy()
        Sigma_post = np.cov(post_samples.detach().numpy(), rowvar=False)
        prec_post = np.linalg.inv(Sigma_post)

        t1 = np.log(np.linalg.det(prec_post) / np.linalg.det(prec_analytical))
        t2 = np.trace(Sigma_post @ prec_analytical)
        t3 = np.matmul(mu_analytical - mu_post, np.matmul(Sigma_post, np.transpose(mu_analytical - mu_post, 0)))

        return 0.5 * (t1 + t2 - d + t3)

    def kl_div_lik(self, lik_samples):
        # post_samples = posterior_samples_iter[1]

        d = lik_samples.size()[1]

        mu_analytical = self.model.loc.numpy()
        prec_analytical = self.model.precision_matrix.numpy()

        mu_post = lik_samples.mean(0).detach().numpy()
        Sigma_post = np.cov(lik_samples.detach().numpy(), rowvar=False)
        prec_post = np.linalg.inv(Sigma_post)

        t1 = np.log(np.linalg.det(prec_post) / np.linalg.det(prec_analytical))
        t2 = np.trace(Sigma_post @ prec_analytical)
        t3 = np.matmul(mu_analytical - mu_post, np.matmul(Sigma_post, np.transpose(mu_analytical - mu_post, 0)))

        return 0.5 * (t1 + t2 - d + t3)

    # TODO add KL div for likelihood as well!


class ConjugateMultivariateNormalSummayStats(ConjugateMultivariateNormal):

    def __init__(self, model, prior, N, x_dim):
        super().__init__(model, prior, N, x_dim)

    # over-write model sim function
    def model_sim(self, theta):
        """
        Simulate from model for a given theta

        :param theta: Matrix of size n x dim_theta
        :return: x_samples: Matrix of size x x S(x) with samples from the model
        """

        n = theta.shape[0]

        x_samples = torch.zeros(n, self.N, self.x_dim)

        for j in range(theta.shape[0]):
            model_tmp = MultivariateNormal(theta[j, :], self.model.covariance_matrix)
            x_samples[j, :, :] = model_tmp.rsample(sample_shape=(self.N,))

        print(x_samples.shape)

        return func.calc_summary_stats(x_samples)
