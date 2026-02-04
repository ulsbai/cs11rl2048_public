import sys
import time
import random
import numpy as np
import os, json, pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import _2048
from dqn import DQNAgent
from stat_tracking import _2048StatTracker


# Classes made by nonthesis person that I rewrote
class ConvBlock(nn.Module):
  '''
    This is one convolutional block layer for the convolutional neural network.
    This has four sub-convolutional layers with kernel sizes 1, 2, 3, and 4.
    The outputs of the sub-convolutional layers are concatenated along the channel dimension.
  '''
  
  def __init__(self, in_channels, out_channels):
    '''
      Constructor.
      in_channels and out_channels are the number of input channels and total output channels (after concatenation), respectively.
      Since the output comes from four outputs concatenated, out_channels must be a multiple of 4.
    '''
    
    super(ConvBlock, self).__init__()
    
    assert out_channels % 4 == 0
    
    self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, padding='same')
    self.conv2 = nn.Conv2d(in_channels, out_channels//4, kernel_size=2, padding='same')
    self.conv3 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding='same')
    self.conv4 = nn.Conv2d(in_channels, out_channels//4, kernel_size=4, padding='same')
  
  def forward(self, x):
    '''Forward function. Applies the convolutional block layer only to input x'''
    
    return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], dim=-3)

def _2048OneHotConvNN_do_one_hot(x):
  '''
    Applies the one-hot encoding on input x and permutes dimensions so that the one-hot dimension is
    behind the tile position dimensions, and reshapes preparing for the convolutional neural network.
  '''
  
  x = x.view(x.shape[0], 4,4) if x.ndim==2 else x.view(4,4)
  x = F.one_hot(x, num_classes=12).float() # Do one-hot encoding
  x = x.permute(-1,-3,-2) if x.ndim==3 else x.permute(0,-1,-3,-2) # put one-hot dim behind grid dims
  # At the end the x.shape is (batch_size, one_hot_class, grid_height, grid_width)
  return x

class _2048OneHotConvNN(nn.Module):
  '''Convolutional neural network for 2048 which ended up working well.'''
  
  def __init__(self):
    '''Constructor.'''
    
    super(_2048OneHotConvNN, self).__init__()
    
    self.convblock1 = ConvBlock(12, 128)
    self.convblock2 = ConvBlock(128, 256)
    self.convblock3 = ConvBlock(256, 256)
    self.dense1 = nn.Linear(256*16, 512)
    self.dense2 = nn.Linear(512, 4)
  
  def forward(self, x):
    '''Forward function, applies the one-hot encoding, then the convolutional neural network to input x'''
    
    x = _2048OneHotConvNN_do_one_hot(x)
    x = self.forward_from_one_hot(x)
    return x
  
  def forward_from_one_hot(self, x):
    '''
      Applies the convolutional neural network only to input x,
      which is assumed to already have been encoded using the one-hot encoding.
    '''
    x = F.relu(self.convblock1(x))
    x = F.relu(self.convblock2(x))
    x = F.relu(self.convblock3(x))
    x = x.flatten(start_dim=-3)
    x = F.relu(self.dense1(x))
    x = F.relu(self.dense2(x))
    return x

ONEHOT_MODELS = (_2048OneHotConvNN,)

def state_to_tensor(state):
  '''Flattens the state np array and converts it to a torch tensor'''
  
  obs = torch.as_tensor(state.flatten(), dtype = torch.int64 if MODEL in ONEHOT_MODELS else torch.float32)
  #print(obs)
  return obs

class Train2048DQN:
  '''Class for training a DQN for 2048.'''
  
  def __init__(self, agent, env, learn_episodes, sync_learns, learngsz, epsilon=0, epsilon_decay=0, epsilon_decay_type='linear', min_epsilon=0, whet_action_masking=False):
    '''
      Constructor
      Takes as input the agent, environment, and a bunch of hyperparameters.
    '''
    
    self.agent = agent
    self.env = env
    
    self.obs = state_to_tensor(self.env.reset()[0])
    self.whet_action_masking = whet_action_masking
    
    self.learn_episodes = learn_episodes
    self.episodes_since_learn = 0
    self.sync_learns = sync_learns
    self.learns_since_sync = 0
    self.learngsz = learngsz
    
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_decay_type = epsilon_decay_type
    self.min_epsilon = min_epsilon
    
    self.stats = _2048StatTracker()
  
  def step(self):
    '''
      Called repeatedly for training. Performs one step of the game.
      Chooses actions using an epsilon-greedy policy
      If the game is over, the method will automatically reset to the next game.
      Once in a while, the method will automatically learn from the experiences gathered.
      Once in a longer while, the method will automatically synchronize the policy and target
      networks (copying the parameters from the policy network to the target network).
    '''
    
    if self.epsilon and random.random()<self.epsilon:
      if self.whet_action_masking:
        action = random.choice(_2048.ACTION_SPACE[self.env.action_masks()])
      else:
        action = random.randint(0, _2048.ACTION_SPACE.size-1)
    
    elif self.whet_action_masking:
      action = self.agent.predict(self.obs, action_masks=self.env.action_masks())
    else:
      action = self.agent.predict(self.obs)
    
    new_state, reward, terminated, truncated, _ = self.env.step(action)
    
    # update agent
    new_obs = state_to_tensor(new_state)
    self.agent.update(self.obs, action, new_obs, reward, terminated)
    self.obs = new_obs
        
    self.stats.step(reward=reward, epsilon=self.epsilon)
    
    if terminated or truncated:
      #print('finished episode!')
      maxtile = np.max(self.env.game.get_grid())
      self.obs = state_to_tensor(self.env.reset()[0])
      self.stats.episode(maxtile=maxtile)
      
      # Decrement epsilon every episode
      if self.epsilon_decay_type == 'linear':
        self.epsilon = max(self.min_epsilon, self.epsilon-self.epsilon_decay)
      elif self.epsilon_decay_type == 'exponential':
        self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)
      else:
        raise ValueError(f"epsilon_decay_type must be either 'linear' or 'exponential' but found {self.epsilon_decay_type!r}")
      
      self.episodes_since_learn += 1
      if self.episodes_since_learn >= self.learn_episodes:
        self.episodes_since_learn = 0
        for k in range(self.learngsz):
          loss = self.agent.learn()
          self.stats.learn(loss)
        self.learns_since_sync += 1
        if self.learns_since_sync >= self.sync_learns:
          self.agent.sync_nets()

if __name__ == '__main__':  
  # Hyperparameters
  ENV = _2048._2048Env2_5
  MODEL = _2048OneHotConvNN
  DISCOUNT = 0.99
  BATCH_SIZE = 1000
  BUFFER_SIZE = 50_000
  LEARNING_RATE = 3e-4
  TRUNCATE = 10_000
  LOG2 = MODEL in ONEHOT_MODELS
  LEARN_INTERVAL = 10
  SYNC_LEARNS = 12
  N_EPISODES = 50_000_000 # Some big number so it will never stop training till I tell it to
  EPSILON = 0.9
  MIN_EPSILON = 0.002
  EPSILON_DECAY = 0.999_99
  EPSILON_DECAY_TYPE = 'exponential'
  WHET_ACTION_MASKING = True
  SAVE_EPISODES = 1000
  #CLEAR_EPISODES = 20
  PRINT_EPISODES = 100
  LEARNGSZ = 1
  
  HYPERPARAMETERS_NAME = f"model_{ENV=!r}_{MODEL=!r}_{DISCOUNT=!r}_{BATCH_SIZE=!r}_{BUFFER_SIZE=!r}_{LEARNING_RATE=!r}_{TRUNCATE=!r}_{LOG2=!r}_{LEARN_INTERVAL=!r}_{SYNC_LEARNS=!r}_{EPSILON=!r}_{EPSILON_DECAY=!r}_{EPSILON_DECAY_TYPE=!r}_{MIN_EPSILON=!r}_{WHET_ACTION_MASKING=!r}_{N_EPISODES=!r}_{SAVE_EPISODES=!r}_{PRINT_EPISODES=!r}_{LEARNGSZ=!r}"
  HASH = str(random.randint(0,10**19-1)) #str(hash(HYPERPARAMETERS_NAME))
  TARGET_NET_FILEPATH = os.path.join("pth_models", "target_net_" + HASH + ".pth")
  POLICY_NET_FILEPATH = os.path.join("pth_models", "policy_net_" + HASH + ".pth")
  TRANSITIONS_FILEPATH = os.path.join("transition_buffers", "transitions_" + HASH + ".pickle")
  HYPERPARAMETERS_FILEPATH = os.path.join("hyperparameters", "hyperparameters_" + HASH + ".txt")
  
  # Make sure the required directories exist
  for directory in ("pth_models", "transition_buffers", "hyperparameters"):
    if not os.path.isdir(directory):
      if os.path.exists(directory):
        raise FileExistsError(f"Cannot create directory '{directory}' for saving progress because '{directory}' is a file")
      
      print(f"Creating directory '{directory}' for saving progress")
      os.mkdir(directory)
  
  policy_net = MODEL()
  policy_net.train()  
  target_net = MODEL()
  target_net.eval()
  agent = DQNAgent(policy_net=policy_net, target_net=target_net, discount=DISCOUNT, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, optimizer=optim.Adam(policy_net.parameters(), lr=LEARNING_RATE), state_dtype=torch.long if MODEL in ONEHOT_MODELS else torch.float32)
  env = ENV(truncate=TRUNCATE, log2=LOG2)
  train = Train2048DQN(agent=agent, env=env, learn_episodes=LEARN_INTERVAL, sync_learns=SYNC_LEARNS, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_decay_type=EPSILON_DECAY_TYPE, min_epsilon=MIN_EPSILON, whet_action_masking=WHET_ACTION_MASKING, learngsz=LEARNGSZ)
  
  def save_model(stats):
    '''
      Called several times whenever saving progress is necessary.
      Saves the following things, in order:
      - the target_net, via pytorch
      - the policy_net, via pytorch
      - the experience buffer, via pickle
      - the hyperparameters and stats, via json and some wierd concatenating of hyperparameter strings done earlier
    '''
    
    print()
    print("Saving state...")
    
    # Target net
    torch.save(target_net.state_dict(), TARGET_NET_FILEPATH)
    print("Target net saved to file", TARGET_NET_FILEPATH)
    
    # Policy net
    torch.save(policy_net.state_dict(), POLICY_NET_FILEPATH)
    print("Policy net saved to file", POLICY_NET_FILEPATH)
    
    # Transition buffer
    with open(TRANSITIONS_FILEPATH, 'wb') as f:
      pickle.dump(agent.transitions, f)
    print("Transition buffer saved to file", TRANSITIONS_FILEPATH)
    
    # Hyperparameters
    hyperparameters = {'HYPERPARAMETERS_NAME': HYPERPARAMETERS_NAME, 'stats': str(stats)}
    with open(HYPERPARAMETERS_FILEPATH, 'w') as f:
      json.dump(hyperparameters, f)
    print("Hyperparameters saved to file", HYPERPARAMETERS_FILEPATH)
    
    print("Finished saving state.")
    print()
  
  # train
  print("Training...")
  save_model(train.stats) # Start by saving model to test model saving
  
  ep = train.stats.get_ep_n()
  
  try:
    while ep < N_EPISODES:
      train.step()
      new_ep = train.stats.get_ep_n()
      
      if ep<new_ep:
        #print('Episode', ep)
        
        if new_ep%PRINT_EPISODES == 0:
          print(train.stats)
          train.stats.clear()
        
        if new_ep%SAVE_EPISODES == 0:
          save_model(train.stats)
        
        #if new_ep%CLEAR_EPISODES == 0:
          #train.stats.clear()
          #print()
          #print('Cleared')
          #print()
      
      ep = new_ep
  
  except KeyboardInterrupt:
    save_model(train.stats)
    raise
  
  print("Finished training!")
  save_model(train.stats)
