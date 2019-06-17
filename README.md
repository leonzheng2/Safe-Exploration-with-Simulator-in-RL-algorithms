# Swimmer Task: OpenAI Gym implementation
Contained in `src/openai/`.

## Requirements
Install OpenAI Gym in Python 3. MuJoCo is needed if we want to use [Swimmer-v2 environment](https://gym.openai.com/envs/Swimmer-v2/) for comparison purpose.

## Using LeonSwimmer environment
In order to use Remy's implementation of Swimmer task written with OpenAI Gym environment interface, please follow steps in `gym_swimmer/` folder.

This is necessary if we want to use other implementation of RL algorithms which are compatible with OpenAI Gym framework, such as [OpenAI baselines algorithms](https://github.com/openai/baselines).

Otherwise, we can just import the Environment class to have easy manipulation of parameters.

## Runing an experiment
Run `python src/openai/plot.py` to run an ARS training on the LeonSwimmer environment, and obtain the learning curve over several seeds.

# Swimmer Task: RLGlue Implementation

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

## Build
Run `make` in `src/` directory.

## Project content

### RLGlue Agent
Contained in `src/agent/`.

Written in Python. Implementation of ARS algorithm, adapted to the RLGlue framework.

### RLGlue Environment
Contained in `src/environment/`.

Written in C++. Using the model from Remy Coulom's thesis: https://www.remi-coulom.fr/Thesis/

### RLGlue Experiment
Contained in `src/experiment/`.

Written in C++. Printing rewards after each policy update. Run `./SwimmerExperiment N_IT` with `N_IT` the number of iterations for the ARS algorithm.

### Parameters
Please change the parameters in `src/parameters.txt`.
