# Safe Exploration with CACLA

This directory contains:

* implementation of CACLA algorithm, based on [Reinforcement Learningin Continuous Action Spaces](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.75.7658&rep=rep1&type=pdf)
* implementation of Safe Exploration with CACLA
* experiences showing the effectiveness of the validity of Safe Exploration

## Implementation of CACLA

### Without Safe Exploration

See classes in `cacla_agent.py`.

### With Safe Exploration

See classes in `cacla_safe_agent.py`.

## Solving LQR

### Environment

The OpenAI Gym environment of LQR can be found in `envs/gym_lqr/`.

### Testing the implementation of CACLA for solving LQR

To test if CACLA can solve easy instances of LQR, run the script `lqr_experiment.py`.

### Using Safe Exploration with CACLA on LQR

Implementation of the agent using Safe Exploration with CACLA on LQR can be found in the file `cacla_safe_agent.py`.

To run experiment for comparing CACLA on easy LQR task with and without Safe Exploration, run the script `safe_exploration_lqr.py`.

## Solving Swimmer Task

### Environment

The OpenAI Gym environment of Coulom's Swimmer task can be found in `envs/gym_swimmer/`. For using MuJoCo Swimmer, use the environment `Swimmer-v2`.

### Testing the implementation of CACLA for solving Swimmer

For solving the Coulom's Swimmer task, run the script `swimmer_experiment.py`.

For solving the MuJoCo tasks, run the script `mujoco_experiement.py`.