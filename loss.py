import torch
import torch.nn as nn
from torch.autograd import Variable


class Loss(object):
    """ Base class for encapsulation of the loss functions.
    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.
    Attributes:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.  Implementation depends on individual
            sub-classes.
        acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the loss of multiple batches.  Implementation depends on individual
            sub-classes.
    """

    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.
        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.
        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """ Evaluate and accumulate loss given outputs and expected results.
        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.
        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()

    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()


class PolicyGradientLoss(Loss):
    """ Compute loss from observation, actions, rewards
    Args:
        gamma (float):
        reward_to_go (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        normalize_advantages (bool, optional): index of masked token, i.e. weight[mask] = 0.
        nn_baseline (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """
    _NAME = "Policy Gradient Loss"

    def __init__(self, gamma, reward_to_go=True, normalize_advantages=True, nn_baseline=False, weight=None, mask=None,
                 size_average=True):
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        super(PolicyGradientLoss, self).__init__(
            self._NAME,
            nn.NLLLoss(weight=weight, size_average=size_average,
                       reduce=False))  # Note here. old version has no parameter reduce=True
        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.normalize_advantages = normalize_advantages
        self.nn_baseline = nn_baseline

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data[0]
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, actions, rewards):
        """Compute loss from batch (observation, actions, rewards)
        Args
            outputs (Variable): Policy network output (SoftMax probabilities). Variable with shape [episode_step * batch, action_dim]
            actions (list): Selected actions. Each element is LongTensor with shape [episode_step]
            rewards (list): Reward for each step. Each element is FloatTensor with [episode_step]
        """
        # concatenate batch of episodes
        concatenate_actions = torch.cat(actions)
        concatenate_rewards = torch.cat(rewards)

        # compute total discounted reward
        if self.reward_to_go:
            q_values = torch.zeros(concatenate_rewards.shape)
            index = 0
            for path_rewards in rewards:
                total_step = path_rewards.shape[0]
                steps = torch.arange(0, total_step)
                discount = torch.pow(self.gamma, steps)
                for t in range(total_step):
                    q = torch.sum(discount[:(total_step - t)] * path_rewards[t:])
                    q_values[index] = q
                    index += 1
        else:
            q_values = list()
            for path_rewards in rewards:
                total_step = path_rewards.shape[0]
                steps = torch.arange(0, total_step)
                q = torch.sum(torch.pow(self.gamma, steps) * path_rewards)
                q_values.append(q.expand(total_step))
            q_values = torch.cat(q_values)

        if self.nn_baseline:
            # TODO
            advantages = q_values
        else:
            advantages = q_values

        if self.normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

        if self.nn_baseline:
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            #
            # Fit it to the current batch in order to use for the next iteration. Use the
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            pass

        advantages = Variable(advantages, requires_grad=True)
        concatenate_actions = Variable(concatenate_actions)

        # Compute Loss (advantages, concatenate_actions, concatenate_outputs)
        outputs = torch.log(outputs)
        loss = self.criterion(outputs, concatenate_actions)  # Combine log_softmax with nll_loss (cross_entropy loss)
        loss = torch.mul(loss, advantages)
        loss = torch.mean(loss)
        self.acc_loss += loss
        self.norm_term += 1
