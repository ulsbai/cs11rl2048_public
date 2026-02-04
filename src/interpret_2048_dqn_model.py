import math
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
  
import torch
from torch.utils.benchmark import Timer

from captum.attr import Lime, FeaturePermutation
#from scipy.optimize import curve_fit

import _2048
from dqn import DQNAgent
import train_2048_dqn_model as train
from train_2048_dqn_model import state_to_tensor


LOG2_4096 = round(math.log2(4096))


def collect_obss(agent, env, n_episodes):
  obss = []
  
  for i in range(n_episodes):
    print(f'Episode {i} / {n_episodes}')
    
    done = False
    obs = state_to_tensor(env.reset()[0])
    obss.append(obs.unsqueeze(0))  # unsqueeze batch dimension
    
    while not done:
      action = agent.predict(obs, action_masks=env.action_masks())
      new_state, reward, terminated, truncated, _ = env.step(action)
      toobig = np.any(new_state >= LOG2_4096)
      done = terminated or truncated or toobig
      if not toobig:
        obs = state_to_tensor(new_state)
        obss.append(obs.unsqueeze(0))  # unsqueeze batch dimension
  
  return torch.cat(obss, dim=0)  # cat along batch dimension

def random_obss(batch_size):
  MAXTILE = 2048
  MAXTILE_LOG2 = round(math.log2(MAXTILE))
  N_TILES = MAXTILE_LOG2+1
  assert N_TILES==12
  
  obss = torch.randint(N_TILES, size=(batch_size,16))
  return obss


# Symmetry Insanity

def normfac(out1, out2):
  return (out1.mean(dim=-1) + out2.mean(dim=-1)) / 2

def rmse(out1, out2):
  return torch.sqrt(torch.mean((out1-out2)**2, axis=-1))

def adm(out1, out2):  # Abs difference mean
  return torch.abs(torch.mean(out1, dim=-1) - torch.mean(out2, dim=-1))

def admax(out1, out2):  # Abs different max
  return torch.abs(out1.max(dim=-1).values - out2.max(dim=-1).values)

SYMMETRIES = [
  (lambda x: x, (0,1,2,3)),  # e
  (lambda x: x.flip([-1]), (0,3,2,1)),  # f
  (lambda x: x.flip([-2]), (2,1,0,3)),  # fr2
  (lambda x: x.flip([-2,-1]), (2,3,0,1)),  # r2
  (lambda x: x.transpose(-2,-1), (3,2,1,0)),  # fr3
  (lambda x: x.transpose(-2,-1).flip([-1]), (1,2,3,0)),  # r
  (lambda x: x.transpose(-2,-1).flip([-2]), (3,0,1,2)),  # r3
  (lambda x: x.transpose(-2,-1).flip([-2,-1]), (1,0,3,2)),  # fr
]


SYMMETRY_NAMES = ['e', 'f', 'fr2', 'r2', 'fr3', 'r', 'r3', 'fr']

def symmetry_insanities(model, obss, outputs):  # where outputs = model(obss)  
  rmse_insan = []
  norm_rmse_insan = []
  adm_insan = []
  norm_adm_insan = []
  admax_insan = []
  norm_admax_insan = []
  
  for insym, outperm in SYMMETRIES:
    out1 = model( insym(obss.view(obss.shape[0], 4, 4)).reshape(obss.shape) )
    out2 = outputs.gather(-1, torch.tensor([outperm]).expand(outputs.shape))
    
    f = normfac(out1, out2)
    rmse_insan.append( (rmse_ := rmse(out1, out2)) .unsqueeze(0) )  # On each one, unsqueeze symmetry dimension
    norm_rmse_insan.append( (rmse_ / f) .unsqueeze(0) )
    adm_insan.append( (adm_ := adm(out1, out2)) .unsqueeze(0) )
    norm_adm_insan.append( (adm_ / f) .unsqueeze(0) )
    admax_insan.append( (admax_ := admax(out1, out2)) .unsqueeze(0) )
    norm_admax_insan.append( (admax_ / f) .unsqueeze(0) )
  
  rmse_insan = torch.cat(rmse_insan, dim=0)
  norm_rmse_insan = torch.cat(norm_rmse_insan, dim=0)
  adm_insan = torch.cat(adm_insan, dim=0)
  norm_adm_insan = torch.cat(norm_adm_insan, dim=0)
  admax_insan = torch.cat(admax_insan, dim=0)
  norm_admax_insan = torch.cat(norm_admax_insan, dim=0)
  
  return rmse_insan, norm_rmse_insan, adm_insan, norm_adm_insan, admax_insan, norm_admax_insan

def plot_symins(model, obss, outputs):
  with torch.no_grad():
    print('Finding symmetry insanities...')
    rmse_insan, norm_rmse_insan, adm_insan, norm_adm_insan, admax_insan, norm_admax_insan = symmetry_insanities(model, obss, outputs)
    print('Done finding symmetry insanities.')
  
  fig, ax = plt.subplots()
  #fig.suptitle('Symmetry Insanities')
  fig.tight_layout()
  
  ax.set_title('Symmetry Insanities')
  ax.boxplot(rmse_insan.t().numpy())
  ax.set_xticklabels(SYMMETRY_NAMES)
  ax.set_xlabel('Symmetry name')
  ax.set_ylabel('RMSE insanity')
  '''
  ax_norm_rmse.set_title('norm_rmse')
  ax_norm_rmse.boxplot(norm_rmse_insan.t().numpy())
  ax_norm_rmse.set_xticklabels(SYMMETRY_NAMES)
  ax_norm_rmse.set_xlabel('Symmetry name')
  ax_norm_rmse.set_ylabel('norm_rmse insanity')
  
  ax_adm.set_title('adm')
  ax_adm.boxplot(adm_insan.t().numpy())
  ax_adm.set_xticklabels(SYMMETRY_NAMES)
  ax_adm.set_xlabel('Symmetry name')
  ax_adm.set_ylabel('adm insanity')
  
  ax_norm_adm.set_title('norm_adm')
  ax_norm_adm.boxplot(norm_adm_insan.t().numpy())
  ax_norm_adm.set_xticklabels(SYMMETRY_NAMES)
  ax_norm_adm.set_xlabel('Symmetry name')
  ax_norm_adm.set_ylabel('norm_adm insanity')
  
  ax_admax.set_title('admax')
  ax_admax.boxplot(admax_insan.t().numpy())
  ax_admax.set_xticklabels(SYMMETRY_NAMES)
  ax_admax.set_xlabel('Symmetry name')
  ax_admax.set_ylabel('admax insanity')
  
  ax_norm_admax.set_title('norm_admax')
  ax_norm_admax.boxplot(norm_admax_insan.t().numpy())
  ax_norm_admax.set_xticklabels(SYMMETRY_NAMES)
  ax_norm_admax.set_xlabel('Symmetry name')
  ax_norm_admax.set_ylabel('norm_admax insanity')
  '''
  plt.show()


# Lime

PARTIAL_ATTRS_FILE = 'partial_attrs.pth'

def run_lime_on_lots_of_things(model, obss):
  lime = Lime(model)
  attrs = []
  
  for i in range(obss.shape[0]):
    print(f'Obs {i} / {obss.shape[0]}\r')
    
    obs = obss[i,:]
    attr = []
    for a in _2048.ACTION_SPACE:
      attr.append(lime.attribute(obs.unsqueeze(0), target=int(a), n_samples=100).squeeze(0))
    attrs.append(torch.stack(attr, dim=0))
    
    if i%100==0:
      print(f'Saving partial attrs to file {PARTIAL_ATTRS_FILE}, {i=}...')
      torch.save(torch.stack(attrs, dim=0), PARTIAL_ATTRS_FILE)
    
  return torch.stack(attrs, dim=0)
  # shape is (batch_size, target_action=4, grid_flattened=16)

def benchmark_lime_main(model, agent, env):
  print('Collecting obs...')
  with torch.no_grad():
    obss = collect_obss(agent, env, n_episodes=1)
  obs = obss[torch.randint(obss.shape[0], (1,)), :]
  print('Done collecting obs.')
  
  device = torch.device('mps')
  
  print('Benchmarking lime...')
  lime_cpu = Lime(model)
  m_cpu = Timer("f()", globals = {"f": lambda: lime_cpu.attribute(obs, target=0, n_samples=100)}).blocked_autorange(min_run_time=30)
  print('Done on CPU, doing GPU')
  model.to(device)
  lime_gpu = Lime(model)
  obs_gpu = obs.to(device)
  m_gpu = Timer("f()", globals = {"f": lambda: lime_gpu.attribute(obs_gpu, target=0, n_samples=100)}).blocked_autorange(min_run_time=30)
  print('On CPU:')
  print(m_cpu)
  print('On GPU:')
  print(m_gpu)
  print('Done benchmarking lime.')

def plot_lime(model, obss):
  print('Running lime...')
  attrs = run_lime_on_lots_of_things(model, obss).numpy()
  attrs = attrs.mean(axis=1)
  print('Done running lime.')
    
  rxs, npxs = range(12), np.arange(12)
  
  labels = ['empty' if n==0 else str(1<<n) for n in rxs]
  bylabel = [attrs[obss==i] for i in rxs]
  means = np.array([x.mean() for x in bylabel])
  xrange = npxs[:-3]
  meansrange = means[:xrange.size]
  means_quadfit = np.polynomial.Polynomial.fit(xrange, meansrange, deg=2)(npxs)  # Quadratic fit
  
  fig, ax = plt.subplots()
  fig.tight_layout()
  ax.set_title('Lime Attributes VS. Tile Value')
  ax.boxplot(bylabel, tick_labels=labels, showmeans=True, showfliers=False)  # if we show outliers then it goes way out and we cant see the actual plot
  ax.plot(npxs+1, means_quadfit, label='Quadratic fit')
  ax.axhline(0, c='0')
  ax.set_xlabel('Tile value')
  ax.set_ylabel('Attribute')
  fig.legend()
  
  attrs_vs_pos = attrs.mean(axis=0).reshape(4,4)
  fig, ax = plt.subplots()
  fig.tight_layout()
  ax.set_title('Lime Attributes VS. Tile Position')
  im = ax.imshow(attrs_vs_pos, cmap='gray_r')
  cbar = fig.colorbar(im)
  cbar.set_label('Mean attribute')
  
  plt.show()

if __name__ == '__main__':
  ENV = _2048._2048Env2_5
  train.MODEL = MODEL = train._2048OneHotConvNN
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--filepath') # model file
  parser.add_argument('--symins', action='store_true', default=False)
  parser.add_argument('--lime', action='store_true', default=False)
  parser.add_argument('--game-obs', action='store_true', default=False)
  parser.add_argument('--rand-obs', action='store_true', default=False)
  args = parser.parse_args()
  
  assert bool(args.game_obs) ^ bool(args.rand_obs), "You must have one, but not both, of '--game-obs' and '--rand-obs'."
  assert bool(args.symins) ^ bool(args.lime), "You must have one, but not both, of '--symins' and '--lime'."
  
  model = MODEL()
  model.load_state_dict(torch.load(args.filepath))
  model.eval()
  agent = DQNAgent(target_net=model, policy_net=model, discount=None, batch_size=None, buffer_size=None, optimizer=None)
  
  env = ENV(log2=MODEL in train.ONEHOT_MODELS)
  
  with torch.no_grad():    
    
    if args.game_obs:
      print('Collecting real-game observations.')
      obss = collect_obss(agent, env, n_episodes=100)
    else:
      print('Collecting random observations.')
      obss = random_obss(batch_size=100_000)
      
    outputs = model(obss)
    print('Done collecting obss.')
  
  if args.symins:
    plot_symins(model, obss, outputs)
  else:
    plot_lime(model, obss)
