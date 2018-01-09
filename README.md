# Policy Gradient Algorithm
**(This is a pytorch implementation of cs294 homework2)**

### Requirements
To play with this project, you need:
    python >= 3.5
    pytorch >= 0.30 (need reduce parameter for NLLLoss. old version has no parameter "reduce".)
    gym latest
    
## Following introduction explain the implementation detail

### 1 Policy Network (network.py)
We implement a simple feed forward neural network with hidden layers to fit the policy function $\pi$

### 2 Loss Function (loss.py)
The key idea of policy gradient is compute the policy gradient from environment reward samples to update network parameters.
We use current policy network explore in the environment and record observation, action, reward for each step. Compute policy
loss use batch data.

### 3 Train the network (gym_trainer.py)

### 4 Play cart_pole game use pre_trained policy network （car_pole_player.py）