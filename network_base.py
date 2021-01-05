import torch.nn as nn
import torch
import collections
from abc import ABCMeta, abstractmethod


class NetworkBase(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(NetworkBase, self).__init__()
        self.probe_activation = collections.OrderedDict()
        self.grad_data = collections.OrderedDict()

    def load_activations(self):
        """
        Will transfer probed activation to numpy arrays
        :return:
        """
        numpy_activations = collections.OrderedDict()
        for key, item in self.probe_activation.items():
            numpy_activations[key] = item.data.cpu().numpy()
        return numpy_activations

    def reset_activations(self):
        """
        Will remove currently held information
        :return:
        """
        self.probe_activation = collections.OrderedDict()

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'])

    def hook_generator(self, func, name):
        """
        Will generate a hook that will apply "func" to the gradient and save it in self.grad_data[name]
        :param func:
        :param name: name where to put the gradient
        :return:
        """
        def hook(grad):
            self.grad_data[name] = func(grad)

        self.grad_data[name] = 0
        return hook

    @abstractmethod
    def forward(self, x):
        """
        Define forward as required by nn.module
        :param x:
        :return:
        """
        pass

    @abstractmethod
    def loss(self, predictions, targets):
        """
        Define criterion on which the train loop will call .backward().
        Has to return a single value
        :param predictions: List of network outputs : [output1, output2, ..., outputn]
        :param targets:     List of target labels : [label1, label2, ..., labeln]
        :return:
        """
        pass
