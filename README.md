# breakout-dqn

This repository is an implementation of the DeepMind DQN Algorithm for the ATARI Breakout environment from Minh et al.

![example](demos/demo.gif)

## Usage

To train the model, run `python dqn.py --weights [pretrained weights]`. Various hyperparameters can be set in `dqn.py`. Good pretrained weights are provided in the weights directory, but you can also train from scratch. Experiments are saved in the `experiments` directory by default.

## Training Details

The details of the DQN implementation make a huge difference on performance. The following guidelines were helpful in achieving good error:

1. The original paper was ambiguous about frame skipping in the environment. I originally thought the Q-network was fed the past 4 frames (ignoring frame skipping), but in fact it is fed the past 4 observed frames on top of frame skipping, so essentially the current frame, T, T-4, T-8, and T-12 with the default skip size of 4. 
2. Considering loss of life to be the end of an episode for rewards was helpful (i.e. mark any frame on which loss of life occurs as terminal, but don't reset the environment).
3. The large replay memory can be fit into a reasonable GPU or CPU memory (8GB) by storing frames as unsigned integers and by storing the original and subsequent state in the same cache. Essentially, just save all frames once in an array and sample from it as needed. This implementation also supports a memory cache split over multiple devices, but this was ultimately not needed.

![episode-reward](demos/episode-reward.png)
