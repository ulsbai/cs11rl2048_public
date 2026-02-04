import numpy as np
import random
import gymnasium as gym
from typing import Optional
from abc import abstractmethod


GRID_SHAPE = (4,4)
PROB_OF_4 = 0.1
N_ACTIONS = 4
ACTION_SPACE = np.arange(N_ACTIONS)


def merge_func(x):
  return x+1
def _si_func(x):
  return 0 if x==0 else 1<<x

def _swipe_up(grid):
  score_increase = 0
  
  for row in range(grid.shape[0]):
    for col, rowpiece in enumerate(grid[row,:]):
      if rowpiece==0:        
        for j, colpiece in enumerate(grid[(row+1):,col]):
          if colpiece!=0:
            grid[row,col] = colpiece
            grid[row+1+j,col] = 0
            
            for k, piece2 in enumerate(grid[(row+j+2):,col]):
              if piece2 == colpiece:
                new_val = merge_func(colpiece)
                grid[row,col] = new_val
                score_increase += _si_func(new_val)
                grid[row+j+2+k,col] = 0
                break
              
              elif piece2 != 0:
                break
            
            break
      
      else:
        for j, colpiece in enumerate(grid[(row+1):,col]):
          if colpiece==rowpiece:
            new_val = merge_func(colpiece)
            grid[row,col] = new_val
            score_increase += _si_func(new_val)
            grid[row+1+j,col] = 0
            break
          
          elif colpiece!=0:
            break
  
  return grid, score_increase

def _is_up_allowed(grid):
  for col in range(grid.shape[1]):
    prev = None
    zero = False
    
    for piece in grid[:,col]:
      if zero:
        if piece!=0:
          return True
      
      else:
        if prev==piece:
          return True
        
        zero = piece==0
        prev = piece
  
  return False

def _is_terminated(grid):
  for row in range(grid.shape[0]):
    prev = None
    
    for col, piece in enumerate(grid[row,:]):
      if piece==0:
        return False
      
      elif row!=0 and grid[row-1,col]==piece:
        return False
      
      elif prev==piece:
        return False
      
      else:
        prev = piece
  
  return True

class _2048:
  def __init__(self):
    self.grid = np.zeros(GRID_SHAPE, dtype=int)
    self.addrand()
  
  def __str__(self):
    result = '\n'
    grid = self.get_grid()
    size = len(str(np.max(grid)))
    
    for row in grid:
      result += ' '.join(f'{{: >{size}d}}'.format(x) for x in row) + '\n'
    
    return result
  
  def get_log2_grid(self):
    return self.grid
  
  def get_grid(self):
    return np.where(self.grid==0, 0, 1<<self.grid)
  
  def swipe(self, direction): # returns True if you lost, and False otherwise
    if direction == 0: # swipe up
      _, score_increase = _swipe_up(self.grid)
    elif direction == 1: # swipe right
      _, score_increase = _swipe_up(self.grid.T[::-1,:])
    elif direction == 2: # swipe down
      _, score_increase = _swipe_up(self.grid[::-1,:])
    elif direction == 3: # swipe left
      _, score_increase = _swipe_up(self.grid.T)
    else:
      raise ValueError(f"direction must be in range(0,4), but got {direction!r}")
    
    return score_increase
  
  def direction_alloweds(self):
    a = _is_up_allowed(self.grid), \
        _is_up_allowed(self.grid.T[::-1,:]), \
        _is_up_allowed(self.grid[::-1,:]), \
        _is_up_allowed(self.grid.T) \
    
    #b = not np.array_equal( _swipe_up(self.grid_cp())[0], self.grid ), \
      #not np.array_equal( _swipe_up(self.grid_cp().T[::-1,:])[0], self.grid.T[::-1,:] ), \
      #not np.array_equal( _swipe_up(self.grid_cp()[::-1,:])[0], self.grid[::-1,:] ), \
      #not np.array_equal( _swipe_up(self.grid_cp().T)[0], self.grid.T )
    
    #if a!=b:
      #print(f'{a = }, {b = }')
      #print(self)
      #assert False
    return a
  
  def addrand(self):
    zeros = np.argwhere(self.grid==0)
    if zeros.size != 0:
      self.grid[tuple(random.choice(zeros))] = 2 if random.random()<PROB_OF_4 else 1
  
  def move(self, direction): # returns True if you lost, and False otherwise
    score_increase = self.swipe(direction)
    self.addrand()
    
    # Check for lose
    terminated = _is_terminated(self.grid)
    
    #grid_cp = self.grid.copy()
    #terminated2 = np.array_equal( grid_cp, self.grid ) \
      #and np.array_equal( _swipe_up(grid_cp)[0], self.grid ) \
      #and np.array_equal( _swipe_up(grid_cp.T)[0], self.grid.T ) \
      #and np.array_equal( _swipe_up(grid_cp[::-1,:])[0], self.grid[::-1,:] ) \
      #and np.array_equal( _swipe_up(grid_cp.T[::-1,:])[0], self.grid.T[::-1,:] )
    
    #assert terminated==terminated2
    return score_increase, terminated


class _2048Env(gym.Env):
  def __init__(self, truncate=np.inf, log2=False):
    self.truncate = truncate
    self.log2 = log2
    
    # define state space and action space for compatibility with sb3
    self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=GRID_SHAPE)
    self.action_space = gym.spaces.Discrete(4)
  
  def reset(self, seed:Optional[int]=None, options:Optional[int]=None):
    super().reset(seed=seed)
    
    self.game = _2048()
    self.step_n = 0
    return self.game.get_log2_grid() if self.log2 else self.game.get_grid(), {}#info
  
  def step(self, action):
    score_increase, terminated = self.game.move(action)
    
    reward = self.get_reward(score_increase, terminated)
    
    self.step_n += 1
    truncated = self.step_n >= self.truncate
    return self.game.get_log2_grid() if self.log2 else self.game.get_grid(), reward, terminated, truncated, {}#info
  
  def action_masks(self):
    return np.array(self.game.direction_alloweds())
  
  def render(self):
    print(self.game)
  
  @abstractmethod
  def get_reward(self):
    pass


class _2048Env1(_2048Env):  
  def get_reward(self, score_increase, terminated):
    return -10 if terminated else score_increase-0.6

class _2048Env2(_2048Env):  
  def get_reward(self, score_increase, terminated):
    return -100 if terminated else np.count_nonzero(self.game.grid==0)-5

class _2048Env2_5(_2048Env):  
  def get_reward(self, score_increase, terminated):
    return -100 if terminated else \
      score_increase + np.count_nonzero(self.game.grid==0) - 4


ENV3_STRATEGY = np.array([[ 64, 32, 16,  8 ],
                          [ 32, 16,  8,  4 ],
                          [ 16,  8,  4,  2 ],
                          [  8,  4,  2,  1 ]], dtype=np.float32)
ENV3_STRATEGY /= ENV3_STRATEGY.sum()
ENV3_STRAT_VAL_SCALAR = 100
ENV3_INIT_STRAT_VAL = ENV3_STRAT_VAL_SCALAR * np.sum(ENV3_STRATEGY * np.array([[  0,  0,  0,  0],
                                                                               [  0,  0,  2,  0],
                                                                               [  0,  0,  0,  0],
                                                                               [  0,  0,  0,  0]], dtype=np.float32))

class _2048Env3(_2048Env):  
  def reset(self, seed:Optional[int]=None, options:Optional[int]=None):
    self.strat_val = ENV3_INIT_STRAT_VAL
    return super().reset(seed=seed, options=options)
  
  def _get_strat_val(self):
    grid = self.game.get_grid()
    norm_grid = grid / grid.sum()
    masked = norm_grid * ENV3_STRATEGY
    strat_val = masked.sum()
    strat_val *= ENV3_STRAT_VAL_SCALAR
    return strat_val
  
  def get_reward(self, score_increase, terminated):    
    if terminated:
      reward = -100 # Big punishment for losing
    
    else:
      reward = 4*np.count_nonzero(self.game.grid==0) # Reward for number of empty tiles
      
      reward += 0.2*score_increase # Reward for merged tiles
      
      new_strat_val = self._get_strat_val()
      reward += new_strat_val - self.strat_val # Reward for increase in strat val
      self.strat_val = new_strat_val
      #print(f'{self.strat_val = }')
      
      reward -= 4 # Penalty each move to discourage deadlocks
    
    return reward


if __name__ == '__main__':
  DIRECTIONS = {'[A':0,'[C':1,'[B':2,'[D':3}
  
  game = _2048()
  print("Press arrow to indicate swipe direction, then press Enter.")
  
  while True:
    print(game)
    
    while True:
      direction = input()
      
      if direction in DIRECTIONS:
        _, terminated = game.move(DIRECTIONS[direction])
        break
      else:
        print('Unrecognized input. Please try again.')
        continue
    
    if terminated:
      print('YOU LOST! :-(')
      print(game)
      break
