# Swimmer Task: RLGlue Implementation

## RLGlue framework

The framework of RLGlue is explained in [RL-Glue: Language-Independent Software for Reinforcement-Learning Experiments](http://www.jmlr.org/papers/v10/tanner09a.html). 

For details about RLGlue, please read the [documentation](https://sites.google.com/a/rl-community.org/rl-glue/Home/rl-glue).

RLGlue is defines a protocol for reinforcement learning experiments. 

## Requirements

### RLGlue
Download https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/rl-glue-ext/rlglue-3.04.tar.gz and follow instructions.

### C/C++ Codec
Download https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/rl-glue-ext/c-codec-2.0.tar.gz and follow instructions.

### Python Codec

The Codec version provided on rl-glue community website is written in Python2. Please find a version ported in Python3, like https://github.com/steckdenis/rlglue-py3.

Some little modifications are required in the source code: in the file `utils/TaskSpecVRLGLUE3.py`, line 63

```python
# instead of: self.ts = ts
self.ts = ts.decode()
```

## Project content

### RLGlue Agent
Contained in `agent/`.

Written in Python. Implementation of Augmented Random Search algorithm, adapted to the RLGlue framework.

### RLGlue Environment
Contained in `environment/`.

Written in C++. Using the model from Remy Coulom's thesis: https://www.remi-coulom.fr/Thesis/

### RLGlue Experiment
Contained in `experiment/`.

Written in C++. Printing rewards after each policy update. Run `./SwimmerExperiment N_IT` with `N_IT` the number of iterations for the ARS algorithm.

### Parameters
Please change the parameters in `parameters.txt`. We can use for examples the following values.

- `n_seg = 3`: number of Swimmer segments
- `direction = [1.0, 0.]`: direction along which the Swimmer has to swim fast
- `h_global = 0.01`: timestep for environment step integration
- `N = 1`: number of sampled policy perturbations
- `b = 1`: number of selected perturbations for improving the policy
- `H = 1000`: length of a rollout
- `alpha = 0.02`: step size of policy improvement
- `nu = 0.02`: standard deviation of policy perturbations
- `max_u = 5.`: maximum torque in absolute value
- `l_i = 1.`: length of one Swimmer segment
- `k = 10.`: viscosity coefficient
- `m_i = 1.`: mass of one Swimmer segment

## Build and run
Run `make` in this directory to build.

Then, in this directory, run the following command lines for training an ARS agent to learn Swimmer task:
```bash
rl_glue
./environment/SwimmerEnvironment
python agent/SwimmerAgent.py
./experiment/SwimmerExperiment 100000
```
