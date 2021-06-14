# code inspired by https://github.com/gpapamak/snl/blob/master/simulators/hodgkin_huxley.py and
# https://github.com/mackelab/sbi/blob/main/examples/HH_helper_functions.py

import torch
import numpy as np
import math
from torch.distributions.uniform import Uniform
from sbi.utils import BoxUniform

from scipy import stats as spstats

print("test")

import neuron
from neuron import h

print("test")

# create the neuron
h.load_file('stdrun.hoc')
h.load_file('sPY_template')
h('objref IN')
h('IN = new sPY()')
h.celsius = 36

# create electrode
h('objref El')
h('IN.soma El = new IClamp(0.5)')
h('El.del = 0')
h('El.dur = 200')

# set simulation time and initial voltage
h.tstop = 200.0
h.v_init = -70.0

# record voltage
h('objref v_vec')
h('objref t_vec')
h('t_vec = new Vector()')
h('t_vec.indgen(0, tstop, dt)')
h('v_vec = new Vector()')
h('IN.soma v_vec.record(&v(0.5), t_vec)')


class HodgkinHuxley:
    def __init__(self, case, nbr_params, method="not_snpla"):

        params, settings = self.setup(case, nbr_params)

        self.duration = settings[0]
        self.dt = settings[1]
        self.t_on = settings[2]
        self.curr_level = settings[3]

        self.nbr_params = nbr_params
        self.case = case

        self.log_theta_true = params

        # set prior dists from gt parameter values
        # log scale priors from https://arxiv.org/pdf/1805.07226.pdf
        # self.prior = utils.torchutils.BoxUniform(low=torch.as_tensor(self.log_theta_true) - math.log(2),
        #                                       high=torch.as_tensor(self.log_theta_true) + math.log(1.5))
        # self.prior = utils.torchutils.BoxUniform(low=torch.as_tensor([math.log(0.5), math.log(0.1)]),
        #                                         high=torch.as_tensor([math.log(80), math.log(15)]))

        prior_lower = torch.zeros(self.nbr_params)
        prior_higher = torch.zeros(self.nbr_params)

        # get correct interval!
        for i in range(self.nbr_params):

            if case == "sbi":
                prior_lower[i] = math.log(math.exp(self.log_theta_true[i]) * 1 / 2)
                prior_higher[i] = math.log(math.exp(self.log_theta_true[i]) * 3 / 2)
            elif case == "snl":
                prior_lower[i] = self.log_theta_true[i] - math.log(2)
                prior_higher[i] = self.log_theta_true[i] + math.log(1.5)  # was 1.5

            # if self.log_theta_true[i] < 0:
            #    prior_lower[i] = self.log_theta_true[i] * 3 / 2
            #    prior_higher[i] = self.log_theta_true[i] * 1 / 2
            #    #prior_lower[i] = self.log_theta_true[i] - math.log(3)
            #    #prior_higher[i] = self.log_theta_true[i] + math.log(2)
            # else:
            #    prior_lower[i] = self.log_theta_true[i] * 1 / 2
            #    prior_higher[i] = self.log_theta_true[i] * 3 / 2
            #    #prior_lower[i] = self.log_theta_true[i] - math.log(2)
            #    #prior_higher[i] = self.log_theta_true[i] + math.log(3)

        if method == "snpla":
            self.prior = Uniform(low=prior_lower, high=prior_higher)
        else:
            self.prior = BoxUniform(low=prior_lower, high=prior_higher)

    def setup(self, case, nbr_params):

        params = np.zeros(nbr_params)
        settings = np.zeros(4)

        if case == "sbi":

            # gt parameter values for priors
            # for the sbi case https://github.com/mackelab/sbi/blob/main/examples/00_HH_simulator.ipynb

            if nbr_params == 12:

                params[0] = math.log(0.05 * 1000)  # gbar_Na
                params[1] = math.log(0.005 * 1000)  # gbar_K
                params[2] = math.log(0.0001)  # g_leak 0.1 mS/cm2
                params[3] = math.log(53)  # E_Na 53  mV
                params[4] = math.log(107)  # E_K  -107  mV
                params[5] = math.log(70)  # E_leak -70  mV
                params[6] = math.log(0.07)  # gbar_M  0.07  # mS/cm2
                params[7] = math.log(6e2)  # tau_max  6e2 ms
                params[8] = math.log(0.5)
                params[9] = math.log(40)
                params[10] = math.log(60)  # Vt mV
                params[11] = math.log(1)  # sigma 0.1  uA/cm2 # set to 0.5!!

            elif nbr_params == 10:

                params[0] = math.log(0.05 * 1000)  # gbar_Na
                params[1] = math.log(0.005 * 1000)  # gbar_K
                params[2] = math.log(0.0001)  # g_leak 0.1 mS/cm2
                params[3] = math.log(53)  # E_Na 53  mV
                params[4] = math.log(107)  # E_K  -107  mV
                params[5] = math.log(70)  # E_leak -70  mV
                params[6] = math.log(0.07)  # gbar_M  0.07  # mS/cm2
                params[7] = math.log(6e2)  # tau_max  6e2 ms
                params[8] = math.log(60)  # Vt mV
                params[9] = math.log(1)  # sigma 0.1  uA/cm2 # set to 0.5!!

            # sys current parameters
            settings[0] = 200
            settings[1] = 0.025
            settings[2] = 10
            settings[3] = 5e-4

        elif case == "snl":

            if nbr_params == 12:

                params[0] = math.log(0.2 * 1000)  # gbar_Na
                params[1] = math.log(0.05 * 1000)  # gbar_K
                params[2] = math.log(1e-4 * 1000)  # g_leak mS/cm2
                params[3] = math.log(50)  # E_Na   mV
                params[4] = math.log(100)  # E_K    mV
                params[5] = math.log(70)  # E_leak  mV
                params[6] = math.log(7e-5 * 1000)  # gbar_M    # mS/cm2
                params[7] = math.log(1000)  # tau_max   ms
                params[8] = math.log(0.5)
                params[9] = math.log(40)
                params[10] = math.log(60)  # Vt mV
                params[11] = math.log(1)  # sigma  uA/cm2

            elif nbr_params == 10:

                params[0] = math.log(0.2 * 1000)  # gbar_Na
                params[1] = math.log(0.05 * 1000)  # gbar_K
                params[2] = math.log(1e-4 * 1000)  # g_leak mS/cm2
                params[3] = math.log(50)  # E_Na   mV
                params[4] = math.log(100)  # E_K    mV
                params[5] = math.log(70)  # E_leak  mV
                params[6] = math.log(7e-5 * 1000)  # gbar_M    # mS/cm2
                params[7] = math.log(1000)  # tau_max   ms
                params[8] = math.log(60)  # Vt mV
                params[9] = math.log(1)  # sigma  uA/cm2

            # sys current parameters
            settings[0] = 100
            settings[1] = 0.025
            settings[2] = 10
            settings[3] = 5e-4

        # elif case == "Lueckmann17":

        return params, settings

    def simulator(self, params, seed, return_input_current=False):
        """
        Run the simulation for one setting of parameters.
        """

        # parameters
        if self.nbr_params == 12:

            gbar_Na = math.exp(params[0]) / 1000
            gbar_K = math.exp(params[1]) / 1000
            g_leak = math.exp(params[2]) / 1000
            E_Na = math.exp(params[3])
            E_K = -math.exp(params[4])
            E_leak = -math.exp(params[5])
            gbar_M = math.exp(params[6]) / 1000
            tau_max = math.exp(params[7])
            kappa_beta_n_1 = math.exp(params[8])
            kappa_beta_n_2 = math.exp(params[9])
            Vt = -math.exp(params[10])
            sigma = math.exp(params[11])  # sigma

        elif self.nbr_params == 10:

            gbar_Na = math.exp(params[0]) / 1000
            gbar_K = math.exp(params[1]) / 1000
            g_leak = math.exp(params[2]) / 1000
            E_Na = math.exp(params[3])
            E_K = -math.exp(params[4])
            E_leak = -math.exp(params[5])
            gbar_M = math.exp(params[6]) / 1000
            tau_max = math.exp(params[7])
            kappa_beta_n_1 = 0.5
            kappa_beta_n_2 = 40
            Vt = -math.exp(params[8])
            sigma = math.exp(params[9])  # sigma

        # fixed parameters
        C = 1.0  # uF/cm2
        # kappa_beta_n_1 = 0.5
        # kappa_beta_n_2 = 40

        # set parameters
        h.IN.soma[0](0.5).g_pas = g_leak  # g_leak
        h.IN.soma[0](0.5).gnabar_hh2 = gbar_Na  # gbar_Na
        h.IN.soma[0](0.5).gkbar_hh2 = gbar_K  # gbar_K
        h.IN.soma[0](0.5).gkbar_im = gbar_M  # gbar_M
        h.IN.soma[0](0.5).e_pas = E_leak  # E_leak
        h.IN.soma[0](0.5).ena = E_Na  # E_Na
        h.IN.soma[0](0.5).ek = E_K  # E_K
        h.IN.soma[0](0.5).vtraub_hh2 = Vt  # V_T
        h.IN.soma[0](0.5).kbetan1_hh2 = kappa_beta_n_1  # k_betan1
        h.IN.soma[0](0.5).kbetan2_hh2 = kappa_beta_n_2  # k_betan2
        h.taumax_im = tau_max  # tau_max

        if seed is not None:
            rng = np.random.RandomState(seed=seed)
        else:
            rng = np.random.RandomState()

        # set up current injection of noise
        Iinj = rng.normal(0.5, sigma, np.array(h.t_vec).size)
        Iinj_vec = h.Vector(Iinj)
        Iinj_vec.play(h.El._ref_amp, h.t_vec)

        # initialize and run
        neuron.init()
        h.finitialize(h.v_init)
        neuron.run(h.tstop)

        if return_input_current:

            return np.array(h.v_vec), np.array(Iinj)

        else:

            return np.array(h.v_vec)

    def calculate_summary_statistics(self, v):
        """
        Calculate summary statistics
        """

        t = np.array(h.t_vec)
        dt = t[1] - t[0]
        eps = 1.0e-7

        # put everything to -10 that is below -10 or has negative slope
        v_calc_nbr_spikes = np.copy(v)
        ind = np.where(v_calc_nbr_spikes < -10)
        v_calc_nbr_spikes[ind] = -10
        ind = np.where(np.diff(v_calc_nbr_spikes) < 0)
        v_calc_nbr_spikes[ind] = -10

        # remaining negative slopes are at spike peaks
        ind = np.where(np.diff(v_calc_nbr_spikes) < 0)
        spike_times = np.array(t)[ind]
        spike_times_stim = spike_times

        # number of spikes
        if spike_times_stim.shape[0] > 0:
            spike_times_stim = spike_times_stim[
                np.append(1, np.diff(spike_times_stim)) > 0.5
                ]

        # first 2 moments
        m1 = np.mean(v)
        std_v = np.std(v) + eps
        logm2 = np.log(std_v)

        v_standardized = (v - m1) / std_v

        # standardized 3th, 5th and 7th moments
        # std_pw = np.power(np.std(v_on), [3, 5, 7])
        standardized_moments = spstats.moment(v_standardized, [3, 5, 7])

        # log-standardized 4th, 6th and 8th moments
        # std_pw = np.power(np.std(v_on), [4, 6, 8])
        log_moments = np.log(spstats.moment(v_standardized, [4, 6, 8]) + eps)

        moments = np.concatenate((np.array([m1, logm2]), standardized_moments, log_moments,))

        # compute autocorrelations
        n_auto_corr_lags = 10

        auto_corrs = torch.zeros(n_auto_corr_lags)

        # v_tensor = torch.as_tensor(v)
        nbr_samples = v.shape[0]

        # standardize before calc correlations
        # v_tensor = (v_tensor - v_tensor.mean()) / v_tensor.std()

        # v_wo_mean = v - m1

        for i in range(n_auto_corr_lags):
            lag = int(2.5 / dt) * (i + 1)
            # auto_corrs[i] = torch.dot(v_tensor[lag:], v_tensor[:-lag]) / (nbr_samples - 1)
            auto_corrs[i] = np.sum(v_standardized[:-lag] * v_standardized[lag:]) / nbr_samples

        # concatenation of summary statistics
        sum_stats_vec = np.concatenate(
            (
                np.array([spike_times_stim.shape[0]]),
                moments,  # moments
                auto_corrs.numpy(),  # autocorrelations
            )
        )

        return sum_stats_vec
