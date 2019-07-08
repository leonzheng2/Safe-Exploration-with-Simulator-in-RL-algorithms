# LeonSwimmer in OpenAI Gym

## Setup

In the installed gym package, go to `gym/envs/`.

In the `__init__.py` file, add the `register(...)` line of the `register.py` file from this directory.

Then, copy the `swimmer/` directory in `gym/envs/`.

The LeonSwimmer environment's are ready to use.

## Choosing the environment parameters

Change the parameters in `register(...)` from `gym/envs/__init__.py`. The environment is created by using the given environment ID.