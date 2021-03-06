
## Code for: Sequential Neural Posterior and Likelihood Approximation

This repo contains the code for the paper *Sequential Neural Posterior and Likelihood Approximation* arxiv-link. 

The results presented in the **secound** Arxiv version of this paper where generated with the code at tag `preprint2`  

## Computer environment

The models were implemented using `PyTorch` utilizing the packages `nflows` and `sbi`.

The models were trained and run on the `LUNARC` computer system http://www.lunarc.lu.se/, and the results were analysed on a local computer.  

### System settings

|                  | `LUNARC`         |Local computer   | 
|------------------|------------------|-----------------|
| Operating system | `CentOS Linux 7` | `Ubuntu 16.04`  |
| Python version   | `3.7.4`          | `3.7.4`         |
| Package manager  | `pip`            | `conda`         |
| Requirements     | `env_lunarc.txt` | `env_local.yml` |

## Code structure 

- `/algorithms` - source code for the snpla method
- `/util` - source code for some utility functions
- `/mv_gaussian` - source code, run scripts, and notebooks for the MV Gaussian examples
- `/two_moons` - source code, run scripts, and notebooks for the two-moons examples
- `/lotka_volterra` - source code, run scripts, and notebooks for the Lotka-Volterra example
- `/hodgkin_huxley` - source code, run scripts, and notebooks for the Lotka-Volterra example

The code for each experiment is structured as following:

- The files `functions.py` and `CaseStudy.py` contain various classes and functions that defined the model
- The `run_script_"algorithm".py` files are the run scripts
- The notebook `analysis.py` is used to produce all analysis and  plots
- The `*.sh` files in the `/lunarc` folder are the scripts used to run the algorithms on the  `LUNARC` system


## Model simulator for the Hodgkin-Huxley model

We used the `Neuron` software (https://neuron.yale.edu/neuron/) to simulate the Hodgkin-Huxley model. The `Neuron` software was installed on our local computer, and all simulations and calculations for the Hodgkin-Huxley were carried out on our local computer. When simulating the Hodgkin-Huxley model, we utilized the same `Neuron` set up as in *Sequential neural likelihood* (http://proceedings.mlr.press/v89/papamakarios19a.html)   

## Data

The data used for all case studies can be generated from the code.


## How to replicate the results

The results for case study `C` and algorithm `A` are computed by running the scripts `A_main.sh` and the `A_main_h.sh` scripts 
in `/lunarc` folder for case study `C`. The script `A_main_h.sh` will run the hyper-parameter search scheme and the script `A_main.sh`  
will run the algorithm for the different data sets that are considered for case study `C`.   


##  Acknowledgements

The computations were enabled by resources provided by the Swedish National Infrastructure for Computing (SNIC) at LUNARC (http://www.lunarc.lu.se/).
