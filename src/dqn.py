import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
  '''
    This class represent a deep Q-network (DQN) agent
  '''
  
  def __init__(self, policy_net, target_net, discount, batch_size, buffer_size, optimizer, state_dtype=torch.float32):
    '''
      Initializes the DQN
      Takes as input a bunch of hyperparameters and the two input networks policy_net and target_net
    '''
    
    self.discount = discount
    self.batch_size = batch_size
    #self.buffer_size = buffer_size
    
    # Models
    self.policy_net = policy_net
    self.target_net = target_net
    self.sync_nets()
    
    self.optimizer = optimizer
    self.criterion = nn.MSELoss()
    
    self.transitions = deque(maxlen=buffer_size)
    self.state_dtype = state_dtype
  
  def predict(self, obs, action_masks=None):
    '''
      Uses the DQN to greedily predict an action based on observation obs
      If action_masks is specified, it will use action masking
    '''
    
    outputs = self.policy_net(obs)
    
    if action_masks is not None:
      outputs = torch.where(torch.as_tensor(action_masks), outputs, -torch.inf)
    
    action = torch.argmax(outputs).item()
    return action
  
  def update(self, old_obs, action, new_obs, reward, terminated):
    '''
      Adds a transition (old_obs, action, new_obs, reward, terminated) to the transition buffer
    '''
    
    self.transitions.append((old_obs, action, new_obs, reward, int(terminated)))
  
  def learn(self):
    '''
      Samples a batch of size batch_size from the last buffer_size transitions added to the buffer, and learns from them
      Returns the loss
    '''
    
    batch = random.sample(self.transitions, min(self.batch_size, len(self.transitions)))
    old_obss, actions, new_obss, rewards, terminateds = zip(*batch)
    
    old_obss = torch.stack(list(old_obss), dim=0).to(dtype=self.state_dtype)
    actions = torch.tensor(actions, dtype=torch.long)
    new_obss = torch.stack(list(new_obss), dim=0).to(dtype=self.state_dtype)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    terminateds = torch.tensor(terminateds, dtype=torch.float32)
    
    old_q = self.policy_net(old_obss).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(dim=1)  # syntax is torch.Tensor.gather(dim, index)
    with torch.no_grad():
      new_values = self.target_net(new_obss).max(dim=1)[0]
    new_q = rewards +  (1-terminateds) * self.discount * new_values  # if not terminated, add discounted next-state value
    loss = self.criterion(old_q, new_q.detach())
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    return loss.mean().item()
  
  def sync_nets(self):
    '''Copies parameters from the policy_net into the target_net'''
    self.target_net.load_state_dict(self.policy_net.state_dict())
