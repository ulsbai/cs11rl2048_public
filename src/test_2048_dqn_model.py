import time, argparse
import torch

import _2048
from dqn import DQNAgent
import train_2048_dqn_model as train
from stat_tracking import _2048StatTracker

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
  ENV = _2048._2048Env2_5
  train.MODEL = MODEL = train._2048OneHotConvNN
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--filepath')
  parser.add_argument('-r', '--render', action='store_true')
  parser.add_argument('-n', '--num-episodes', type=int, default=100)
  args = parser.parse_args()
  
  model = MODEL()
  model.load_state_dict(torch.load(args.filepath))
  model.eval()
  agent = DQNAgent(target_net=model, policy_net=model, discount=None, batch_size=None, buffer_size=None, optimizer=None)
  
  stats = _2048StatTracker()
  
  # test
  print("Testing...")
  assert MODEL in train.ONEHOT_MODELS
  env = ENV(truncate=10000, log2=MODEL in train.ONEHOT_MODELS)
  
  for _ in range(args.num_episodes):
    done = False
    obs = train.state_to_tensor(env.reset()[0])
    print('Episode', stats.get_ep_n())
    
    if args.render:
      env.render()
    
    while not done:
      action = agent.predict(obs, action_masks=env.action_masks())
      new_state, reward, terminated, truncated, _ = env.step(action)
      stats.step(reward=reward)
      done = terminated or truncated
      
      if args.render:
        env.render()
        time.sleep(1)
        if terminated:
          print('I LOST! :-(')
      
      obs = train.state_to_tensor(new_state)
    
    stats.episode(maxtile=env.game.get_grid().max())
  
  print(stats.to_str_not_training())
