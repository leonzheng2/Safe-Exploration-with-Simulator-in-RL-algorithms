# Research Internship at FLL (April - August 2019)

Work by Léon Zheng (Ecole polytechnique, FLL) under the supervision of Iwane Hidenao (FLL).

## Requirements

Installation of the followings are required:
- Python 3.6
- OpenAI Gym
- MuJoCo
- PyTorch
- Ray
- Eigen

## Swimmer task implementation

The implementation of Swimmer task based on [Remi Coulom's work](https://www.remi-coulom.fr/Publications/Thesis.pdf) is 
implemented in `envs/gym_swimmer/` using OpenAI Gym framework.

For the [RL-Glue](http://www.jmlr.org/papers/volume10/tanner09a/tanner09a.pdf) version of the Swimmer task, see the directory `rlglue/`.

## Safe Exploration for long rollouts algorithms (ARS)

First version of Safe Exploration with _reward constraint_ can be found in the directory `ars/`.

It focuses on the use of [ARS algorithm](https://arxiv.org/pdf/1803.07055.pdf) for solving Swimmer task, with and withour Safe Exploration.

This work helped to stress out the difficulty of using Safe Exploration for long rollouts algorithms.

## Safe Exploration for short rollouts algorithms (CACLA)

Second version of Safe Exploration with _state constraint_ can be found in the directory `cacla/`.

Here, we use easy parameterized instances of Linear Quadratic Regulator to show the validity of the Safe Exploration algorithm.
