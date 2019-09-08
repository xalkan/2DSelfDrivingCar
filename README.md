
# 2D Self Driving Car

A 2D self driving car using Deep Q Networks in pytorch

<img src="https://github.com/xalkan/2DSelfDrivingCar/blob/master/output.gif" />

This is the first project from the course on Artificial Intelligence A-Z after learning about the Q learning algorithm, Markov Decision Processes, Temporal Difference and Deep Q Networks.
The goal is to train this tiny car to learn to move from top left to bottom right in a loop.

#### Stage 1
The tiny car is untrained and moves like an insect. It explores the map and trains to move from starting position to goal.
#### Stage 2
The tiny car is trained to move from top left to bottom right.
### Stage 3
If we draw a random path, it gives up exploration and again exploits itself to train and follow the path. Once, it is trained on the newly created path, it continues to follow it. If we remove the path, it goes back to exploration while following its previous route.

  

## Setting up the Environment

Clone the repo and cd into it.
Create a virtual environment using venv or conda environment
```
# using venv
python -m venv venv

# using conda
conda create --name pytorchenv python=3.5

#activate env on windows
venv/scripts/activate

#activate env on linux
venv/bin/activate
```  

With the latest PyTorch release for Windows, it is much simpler to install pytorch.
Install by going to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and getting the link for your configuration

```
# using pip (python 3.7 windows 10)
pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

```
But this course uses an old version so if the above doesn't work, use

```

conda install -c peterjc123 pytorch-cpu

```

  

Install Kivy

```

pip install docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew

pip install kivy.deps.gstreamer

pip install kivy

```

  

Apart from these, matplotlib and numpy should be installed

    pip install numpy matplotlib

  

Run map.py to confirm successful configuration

    python map.py
