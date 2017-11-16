# OpenAI Gym Environment Wrapper for [Air Hockey Game Simulator](https://github.com/arakhmat/air-hockey)
### Prerequisites
I recommend using Anaconda3 Python distribution. You can download it from: [https://www.anaconda.com/download/](https://www.anaconda.com/download/)
### Installing
1. [Install Air Hockey Game Simulator](https://github.com/arakhmat/air-hockey#installing)
2. Clone the repository and install it as a python module:
```
git clone https://github.com/arakhmat/gym-air-hockey
cd gym-air-hockey
pip install -e .
```
### How to use
Import the environment:
```
import gym
env = gym.make('AirHockey-v0')
```
Import the processor (required if using [keras-rl](https://github.com/matthiasplappert/keras-rl)):
```
import gym_air_hockey
processor = gym_air_hockey.DataProcessor()
```
