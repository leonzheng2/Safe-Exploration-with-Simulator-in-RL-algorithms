# Swimmer Task: OpenAI Gym implementation
Contained in `src/openai/`.

## Requirements
Install OpenAI Gym in Python 3. MuJoCo is needed if we want to use [Swimmer-v2 environment](https://gym.openai.com/envs/Swimmer-v2/) for comparison purpose.

## Using LeonSwimmer environment
In order to use Remy's implementation of Swimmer task written with OpenAI Gym environment interface, please follow steps in `gym-swimmer/` folder.

This is necessary if we want to use other implementation of RL algorithms which are compatible with OpenAI Gym framework, such as [OpenAI baselines algorithms](https://github.com/openai/baselines).

Otherwise, we can just import the Environment class to have easy manipulation of parameters.

## Runing an experiment
Run `python src/openai/plot.py` to run an ARS training on the LeonSwimmer environment, and obtain the learning curve over several seeds.
