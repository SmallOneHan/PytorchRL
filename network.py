import torch
import torch.nn.functional as functional
from torch.autograd import Variable


class PolicyGradientNetwork(object):
    """
    Base class for Policy Gradient Network
    """

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        assert isinstance(model, cls)
        return model

    def make_decision(self, state, discrete, max_probability=False):
        raise NotImplementedError("You should implement 'make_decision' method in sub class")


class FullConnectionNetwork(torch.nn.Module, PolicyGradientNetwork):
    """
    Policy Network for Policy Gradient
    """

    def __init__(self, input_size, output_size, hidden_layers, activation=functional.relu):
        """
        Init network layer
        Args:
        input_size(int): State dim
        output_size(int): Action dim
        hidden_layers(list): hidden layers of this network.
               e.g.[10,5] represent two hidden layers with 10 and 5 units.
        activation(torch.functional): Default activate function is relu
        """
        super(FullConnectionNetwork, self).__init__()
        hidden_layers.insert(0, input_size)
        hidden_layers.append(output_size)
        self.hidden_layers = list()
        for i in range(len(hidden_layers) - 1):
            linear = torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            setattr(self, 'linear_' + str(i), linear)
            self.hidden_layers.append(linear)

        self.activation = activation

    def forward(self, state):
        """
        Apply a feed forward neural network to a input state
        Args:
            state (batch, state_dim): tensor containing the state of the environment representation.
        Returns: output
            - **output** (batch, action_dim): variable containing the action probability distribution
        """
        output = state
        for layer in self.hidden_layers[:-1]:
            output = self.activation(layer(output))

        output = self.hidden_layers[-1](output)
        output = functional.softmax(output, -1)
        return output

    def make_decision(self, state, max_probability=False):
        """
        Make decision by current state
        Args:
            state(Tensor or Variable): environment state shape of (state dimension)
            max_probability(bool): choose max probability action or choose action according to probability distribution
        Returns: decision vector
        """
        if not isinstance(state, Variable):
            if isinstance(state, torch.FloatTensor):
                state = Variable(state)
            else:
                state = Variable(torch.FloatTensor(state))

        output = self.forward(state)

        if max_probability:
            _, decision = output.max(0)
        else:
            decision = torch.multinomial(output, 1).data[0]

        return decision
