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

Written in Python. Implementation of ASR algorithm, adapted to the RLGlue framework.

### RLGlue Environment
Contained in `src/environment/`.

Written in C++. Using the model from Remy Coulom's thesis: https://www.remi-coulom.fr/Thesis/

### RLGlue Experiment
Contained in `src/experiment/`.

Written in C++. Printing rewards after each policy update. Run `./SwimmerExperiment N_IT` with `N_IT` the number of iterations for the ASR algorithm.

### Parameters
Please change the parameters in `src/parameters.txt`.

### OpenAI Gym
Contained in `src/openai/`.

Using the MuJoCo environment and implementing ASR algorithm using OpenAI Gym.