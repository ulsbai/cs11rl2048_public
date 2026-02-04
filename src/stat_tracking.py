import time, datetime
#from torch.utils.tensorboard import SummaryWriter

class StatTracker:
  '''
    The StatTracker class keeps track of various learning progress output
    for it to be printed on the screen.
    The following stats are currently supported:
      Step #
      Episode #
      Avg reward per step and episode
      Avg loss
      Current epsilon
    
    The StatTracker class can also update information to the tensorboard,
    but this feature is disabled for my final submission because the tensorboard
    can be tricky to install and reduces portability.
  '''
  
  def __init__(self, use_tensorboard=False):
    if use_tensorboard:
      # Tensorboard has been disabled because it does not install on some computers
      class UnsupportedTensorboardError(Exception): pass
      raise UnsupportedTensorboardError("Cannot use the tensorboard because the tensorboard does not install on some computers")
    
    class _StatOrganizer:
      pass
    
    self._inst = _StatOrganizer()
    self._cl = _StatOrganizer()
    self._ep = _StatOrganizer()
    self._step = _StatOrganizer()
    
    self.tensorboard = SummaryWriter() if use_tensorboard else None
    
    def inst_clear():
      self._inst.step_n = 0
      self._inst.ep_n = 0
      self._inst.start_time = time.time()
    self._inst.clear = inst_clear
    
    def cl_clear():
      self._cl.sum_rew = 0
      self._cl.sum_loss = 0
      self._cl.learn_cnt = 0
      self._cl.ep_cnt = 0
      self._cl.step_cnt = 0
    self._cl.clear = cl_clear
    
    def ep_clear():
      self._ep.sum_rew = 0
      self._ep.step_cnt = 0
    self._ep.clear = ep_clear
    
    def step_clear():
      self._step.epsilon = None
    self._step.clear = step_clear
    
    self._inst.clear()
    self._cl.clear()
    self._ep.clear()
    self._step.clear()
  
  # Support with statement
  def __enter__(self):
    return self
  
  def __exit__(self):
    self.tensorboard.close()
  
  def __del__(self):
    self.__exit__()
  
  def clear(self):
    '''
      Called each time the progress is printed.
      Clears the sums so that new averages are taken only from the new information.
      Also appends the current stats to the tensorboard if a tensorboard is being used.
    '''
    
    # Update tensorboard
    if self.tensorboard is not None:
      self.tensorboard.add_scalar('Step vs Episode', self.get_step_n() or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('Elapsed Time vs Episode', self.get_elapsed_time() or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('Avg Episode Length', self.get_avg_ep_len() or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('Avg Step Reward', self.get_avg_step_rew() or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('Avg Episode Reward', self.get_avg_ep_rew() or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('Avg Loss', self.get_avg_ep_len() or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('Avg Episode Length', self.get_avg_loss() or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('Epsilon', self.get_curr_epsilon() or 0, self._inst.ep_n)
    
    self._cl.clear()
  
  def learn(self, loss=0):
    '''Called each time the model learns and a loss is computed.'''
    
    self._cl.sum_loss += loss
    self._cl.learn_cnt += 1
  
  def episode(self):
    '''
      Called each episode of training.
      Tracks the length of the episode
      and passes some sums from lower to higher scope.
    '''
    
    self._inst.ep_n += 1
    self._cl.sum_rew += self._ep.sum_rew
    self._cl.step_cnt += self._ep.step_cnt
    self._cl.ep_cnt += 1
    self._ep.clear()
  
  def step(self, reward=0, epsilon=0):
    '''
      Called each step of training.
      reward and epsilon parameters are tracked.
    '''
    
    self._inst.step_n += 1
    self._ep.sum_rew += reward
    self._ep.step_cnt += 1
    self._step.epsilon = epsilon
  
  def get_step_n(self):
    '''Returns the total number of steps of training.'''
    return self._inst.step_n
  
  def get_ep_n(self):
    '''Returns the total number of episodes of training.'''
    return self._inst.ep_n
  
  def get_avg_ep_len(self):
    '''Returns the average episode length since the last call to clear()'''
    
    return None if self._cl.ep_cnt==0 else self._cl.step_cnt / self._cl.ep_cnt
  
  def get_avg_step_rew(self):
    '''Returns the average reward each step since the last call to clear()'''
    return None if self._cl.step_cnt==0 else self._cl.sum_rew / self._cl.step_cnt
  
  def get_avg_ep_rew(self):
    '''Returns the average reward each episode since the last call to clear()'''
    return None if self._cl.ep_cnt==0 else self._cl.sum_rew / self._cl.ep_cnt
  
  def get_avg_loss(self):
    '''Returns the average loss since the last call to clear()'''
    return None if self._cl.learn_cnt==0 else self._cl.sum_loss / self._cl.learn_cnt
  
  def get_curr_epsilon(self):
    '''Returns the last epsilon value that was given to the tracker.'''
    return self._step.epsilon
  
  def get_elapsed_time(self):
    '''Returns the number of seconds that have passed since starting training.'''
    return time.time() - self._inst.start_time
  
  def __str__(self):
    '''Returns a string representation of the current stats.'''
    
    result = '\n'
    result += f"Step: {self.get_step_n()}\n"
    result += f"Episode: {self.get_ep_n()}\n"
    result += f"Elapsed time: {datetime.timedelta(seconds=self.get_elapsed_time())}\n"
    result += f"Avg episode length: {self.get_avg_ep_len() or 0:.3f}\n"
    result += f"Avg step reward: {self.get_avg_step_rew() or 0:.3f}\n"
    result += f"Avg episode reward: {self.get_avg_ep_rew() or 0:.3f}\n"
    result += f"Avg loss: {self.get_avg_loss() or 0:.3f}\n"
    result += f"Epsilon: {self.get_curr_epsilon() or 0:.3f}\n"
    return result
  
  def to_str_not_training(self):
    '''
      Returns a string representation of only the stats
      that are relevant during testing.
    '''
    
    result = '\n'
    result += f"Elapsed time: {datetime.timedelta(seconds=self.get_elapsed_time())}\n"
    result += f"Avg episode length: {self.get_avg_ep_len() or 0:.3f}\n"
    result += f"Avg step reward: {self.get_avg_step_rew() or 0:.3f}\n"
    result += f"Avg episode reward: {self.get_avg_ep_rew() or 0:.3f}\n"
    return result

class _2048StatTracker(StatTracker):
  '''
    Subclass of the StatTracker class which tracks stats for the game of 2048.
    This class tracks the rates at which the model reaches tile values 64 through 4096.
    Also includes all the stats that the base StatTracker class supports.
  '''
  
  def __init__(self):
    super().__init__()
    
    old_cl_clear = self._cl.clear
    def cl_clear():
      old_cl_clear()
      self._cl.cnt64 = 0
      self._cl.cnt128 = 0
      self._cl.cnt256 = 0
      self._cl.cnt512 = 0
      self._cl.cnt1024 = 0
      self._cl.cnt2048 = 0
      self._cl.cnt4096 = 0
    self._cl.clear = cl_clear
    
    self._cl.clear()
  
  def clear(self):
    '''
      Called each time the progress is printed.
      Clears the sums so that new averages are taken only from the new information.
      Also appends the current stats to the tensorboard if a tensorboard is being used.
    '''
    
    super().clear()
    
    # Update tensorboard
    if self.tensorboard is not None:
      self.tensorboard.add_scalar('64 rate', self.get_maxtile_rate(64) or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('128 rate', self.get_maxtile_rate(128) or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('256 rate', self.get_maxtile_rate(256) or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('512 rate', self.get_maxtile_rate(512) or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('1024 rate', self.get_maxtile_rate(1024) or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('2048 rate', self.get_maxtile_rate(2048) or 0, self._inst.ep_n)
      self.tensorboard.add_scalar('4096 rate', self.get_maxtile_rate(4096) or 0, self._inst.ep_n)
  
  def episode(self, maxtile=0):
    '''
      Called each episode of training.
      Takes extra parameter maxtile, specifying the maximum tile
      in the final board position, to allow the class to track tile-reached rates.
    '''
    
    super().episode()
    
    if maxtile >= 64:
      self._cl.cnt64 += 1
    if maxtile >= 128:
      self._cl.cnt128 += 1
    if maxtile >= 256:
      self._cl.cnt256 += 1
    if maxtile >= 512:
      self._cl.cnt512 += 1
    if maxtile >= 1024:
      self._cl.cnt1024 += 1
    if maxtile >= 2048:
      self._cl.cnt2048 += 1
    if maxtile >= 4096:
      self._cl.cnt4096 += 1
  
  def get_maxtile_rate(self, maxtile):
    '''
      Returns the fraction of games in which the model reaches
      or exceeds maxtile since the last call to clear().
      maxtile must be a 2-power between 64 and 4096 inclusive.
    '''
    
    if maxtile == 64:
      return None if self._cl.ep_cnt==0 else self._cl.cnt64 / self._cl.ep_cnt
    if maxtile == 128:
      return None if self._cl.ep_cnt==0 else self._cl.cnt128 / self._cl.ep_cnt
    if maxtile == 256:
      return None if self._cl.ep_cnt==0 else self._cl.cnt256 / self._cl.ep_cnt
    elif maxtile == 512:
      return None if self._cl.ep_cnt==0 else self._cl.cnt512 / self._cl.ep_cnt
    elif maxtile == 1024:
      return None if self._cl.ep_cnt==0 else self._cl.cnt1024 / self._cl.ep_cnt
    elif maxtile == 2048:
      return None if self._cl.ep_cnt==0 else self._cl.cnt2048 / self._cl.ep_cnt
    elif maxtile == 4096:
      return None if self._cl.ep_cnt==0 else self._cl.cnt4096 / self._cl.ep_cnt
    else:
      raise ValueError(f"maxtile must be either 64, 128, 256, 512, 1024, 2048, or 4096, but got {maxtile!r}")
  
  def __str__(self):
    '''Returns a string representation of the current stats.'''
    
    result = super().__str__()
    result += f"64 rate: {self.get_maxtile_rate(64) or 0:.1%}\n"
    result += f"128 rate: {self.get_maxtile_rate(128) or 0:.1%}\n"
    result += f"256 rate: {self.get_maxtile_rate(256) or 0:.1%}\n"
    result += f"512 rate: {self.get_maxtile_rate(512) or 0:.1%}\n"
    result += f"1024 rate: {self.get_maxtile_rate(1024) or 0:.1%}\n"
    result += f"2048 rate: {self.get_maxtile_rate(2048) or 0:.1%}\n"
    result += f"4096 rate: {self.get_maxtile_rate(4096) or 0:.1%}\n"
    return result
  
  def to_str_not_training(self):
    '''
      Returns a string representation of only the stats
      that are relevant during testing.
    '''
    
    result = super().to_str_not_training()
    result += f"64 rate: {self.get_maxtile_rate(64) or 0:.1%}\n"
    result += f"128 rate: {self.get_maxtile_rate(128) or 0:.1%}\n"
    result += f"256 rate: {self.get_maxtile_rate(256) or 0:.1%}\n"
    result += f"512 rate: {self.get_maxtile_rate(512) or 0:.1%}\n"
    result += f"1024 rate: {self.get_maxtile_rate(1024) or 0:.1%}\n"
    result += f"2048 rate: {self.get_maxtile_rate(2048) or 0:.1%}\n"
    result += f"4096 rate: {self.get_maxtile_rate(4096) or 0:.1%}\n"
    return result

if __name__ == '__main__':
  stats = StatTracker()
  print(stats)
  stats.step()
  print(stats)
  stats.step(reward=5, epsilon=0.9)
  print(stats)
  stats.episode()
  print(stats)
  stats.learn(loss=5)
  print(stats)
  stats.clear()
  print(stats)
