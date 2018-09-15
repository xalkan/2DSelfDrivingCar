# 2D Self Driving Car
A 2D self driving car using Deep Q Networks in pytorch

This is the first project from the course on Artificial Intelligence A-Z after learning about the Q learning algorithm, Markov Decision Processes, Temporal Difference and Deep Q Networks.

## Setting up the Environment
Create a new conda environment
```
conda create --name pytorchenv python=3.5
```

With the latest PyTorch release for Windows, it is much simpler to install by just doing
```
conda install -c pytorch pytorch-cpu
```

But this course uses an old version so
```
conda install -c peterjc123 pytorch-cpu
```

Kivy is not on anaconda so install it using
```
pip install docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew
pip install kivy.deps.gstreamer
pip install kivy
```

Apart from these, matplotlib and numpy should be installed.

Run map.py to confirm successful configuration.
