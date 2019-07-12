# Safe Exploration with ARS

This directory contains ARS implementation with and without Safe Exploration. The only environment considerd here 
is the Swimmer task implemented in `envs/gym_swimmer/`.

This version of Safe Exploration considers long rollouts: `H = 1000`, and safety constraints concerns returns of rollouts.
The conclusion of these experiences is that it is more easier and useful to use Safe Exploration on short rollout: `H=1` 
and consider safety constraints on states rather than rewards.

## Description of the files

### Classes
* `parameters.py`: classes for setting parameters of environment, agent
* `experiment.py`: class for creating an experience by setting parameters and output the results in a graph.
* `environment.py`: class for environment
* `ars_agent,py`: class implementing ARS with and without Safe Exploration
* `estimator.py`: class for estimating the real world parameters using simulations
* `database.py`: class for manipulating trajectories

### Script for experiences
* `plot_graph.py`: plot the learning curve of ARS on Swimmer task
* `safe_exploration.py`: Safe Exploration with ARS and constraint on rollout returns
* `state_range.py`: obtain the range of visited states
* `comparison_mujoco.py`: transfering the policy learned in LeonSwimmer to MuJoCo Swimmer
