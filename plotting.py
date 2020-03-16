from visdom import Visdom
import numpy as np

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', port=8050):
        try:
            self.viz = Visdom(port=port)
        except (ConnectionError, ConnectionRefusedError) as e:
            raise ConnectionError("Visdom Server not running, please launch it with `visdom` in the terminal")
    
        self.env = env_name
        self.plots = {}
    
    def clear(self):
        self.plots = {}
        
    def imshow(self, var_name, images):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images)
        else:
            self.viz.images(images, win=self.plots[var_name], env=self.env)

    def plot(self, window_id, variable, title, x, y, xlabel='epochs'):
        if window_id not in self.plots:
            self.plots[window_id] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[variable],
                title=title,
                xlabel=xlabel,
                ylabel=variable
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[window_id], name=variable, update = 'append')


class Logger:
    def __init__(self, name):
        self.name = plottern

        self.value = 0

    def increment(self, value, step=1):
        self.value += value
        self.step += step

    def update(self, value, alpha=0.8):
        self.value = alpha * self.value + (1 - alpha) * value

    def set(self, value):
        self.value = value
    
    def reset(self):
        self.value = 0

    def value(self):
        return self.value