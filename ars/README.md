# Safe Exploration with ARS

This directory contains ARS implementation with and without Safe Exploration. The only environment considerd here 
is the Swimmer task implemented in `envs/gym_swimmer/`.

This version of Safe Exploration considers long rollouts: `H = 1000`, and safety constraints concerns returns of rollouts.
The conclusion of these experiences is that it is more easier and useful to use Safe Exploration on short rollout: `H=1` 
and consider safety constraints on states rather than rewards.

## Requirements
Install OpenAI Gym in Python 3. MuJoCo is needed if we want to use [Swimmer-v2 environment](https://gym.openai.com/envs/Swimmer-v2/) for comparison purpose.

## Using LeonSwimmer environment
In order to use Remy's implementation of Swimmer task written with OpenAI Gym environment interface, please follow steps in `envs/gym-swimmer/` folder.

This is necessary if we want to use other implementation of RL algorithms which are compatible with OpenAI Gym framework, such as [OpenAI baselines algorithms](https://github.com/openai/baselines).

Otherwise, we can just import the Environment class to have easy manipulation of parameters.

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

## Runing experiments

Run the scripts described above after choosing the parameters to make experiments and obtain results.
