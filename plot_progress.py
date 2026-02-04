from pathlib import Path
import re, datetime
import numpy as np
import matplotlib.pyplot as plt, matplotlib.dates as dates
import argparse
import numpy as np

assert __name__ == '__main__'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filepath')
parser.add_argument('-k', '--kernel-size', type=int, default=1)
parser.add_argument('--x-restarts', action='store_true', default=False)
parser.add_argument('--absolute-time', action='store_true', default=False)
args = parser.parse_args()

origin_dt = datetime.datetime.strptime('','')

# Read log file
#txt = Path(args.filepath).read_text()
txt = '\n<NEW_FILE>\n'.join((Path(fp).read_text() for fp in args.filepath.split(',')))  # Support multiple files to be concatenated

# Parse
steps = []
episodes = []
times = []
r64s = []
r128s = []
r256s = []
r512s = []
r1024s = []
r2048s = []
r4096s = []

step_origin = 0
episode_origin = 0
time_origin = datetime.timedelta(0)

barsteps = []
barepisodes = []
bartimes = []

for line in txt.split('\n'):
  if line == '<NEW_FILE>':
    barsteps.append(steps[-1])
    barepisodes.append(episodes[-1])
    bartimes.append(times[-1])
    if args.x_restarts:
      step_origin = steps[-1]
      episode_origin = episodes[-1]
      time_origin = times[-1]-origin_dt
  elif line.startswith('Step:'):
    steps.append(step_origin + int(re.findall(r'\d+', line)[0]))
  elif line.startswith('Episode:'):
    episodes.append(episode_origin + int(re.findall(r'\d+', line)[0]))
  elif line.startswith('Elapsed time'):
    dt = time_origin + datetime.datetime.strptime(re.findall(r'(\d+:\d+:\d+).\d+', line)[0], '%H:%M:%S')
    if 'day' in line:
      days = int(re.findall('(\d+) days?', line)[0])
      dt += datetime.timedelta(days=days)
    times.append(dt)
  elif line.startswith('64 rate'):
    r64s.append(max(0, float(re.findall(r'(-?[.\d]+)%', line)[0])))
  elif line.startswith('128 rate'):
    r128s.append(max(0, float(re.findall(r'(-?[.\d]+)%', line)[0])))
  elif line.startswith('256 rate'):
    r256s.append(max(0, float(re.findall(r'(-?[.\d]+)%', line)[0])))
  elif line.startswith('512 rate'):
    r512s.append(max(0, float(re.findall(r'(-?[.\d]+)%', line)[0])))
  elif line.startswith('1024 rate'):
    r1024s.append(max(0, float(re.findall(r'(-?[.\d]+)%', line)[0])))
  elif line.startswith('2048 rate'):
    r2048s.append(max(0, float(re.findall(r'(-?[.\d]+)%', line)[0])))
  elif line.startswith('4096 rate'):
    r4096s.append(max(0, float(re.findall(r'(-?[.\d]+)%', line)[0])))

if args.absolute_time:
  time_offset : datetime.timedelta = datetime.datetime.now().replace(tzinfo=None) - times[-1]
  times = [t+time_offset for t in times]

steps = np.array(steps)
episodes = np.array(episodes)
sectimes = np.array([(t.replace(tzinfo=None) - origin_dt).total_seconds() for t in times])
f,e,s = steps[1:]-steps[:-1] , episodes[1:]-episodes[:-1] , sectimes[1:]-sectimes[:-1]
fps, eps = f/s, e/s

# Smoothen
if args.kernel_size > 1:
  kernel = np.full(args.kernel_size, 1/args.kernel_size)
  fps = np.convolve(fps, kernel, mode='same')
  eps = np.convolve(eps, kernel, mode='same')
  if r64s: r64s = list(np.convolve(r64s, kernel, mode='same'))
  if r128s: r128s = list(np.convolve(r128s, kernel, mode='same'))
  if r256s: r256s = list(np.convolve(r256s, kernel, mode='same'))
  if r512s: r512s = list(np.convolve(r512s, kernel, mode='same'))
  if r1024s: r1024s = list(np.convolve(r1024s, kernel, mode='same'))
  if r2048s: r2048s = list(np.convolve(r2048s, kernel, mode='same'))
  if r4096s: r4096s = list(np.convolve(r4096s, kernel, mode='same'))

# Make time formatter
timefmt = dates.DateFormatter('%H:%M')

# Plot
fig, ((sax, eax, tax), (fpsax, epsax, badax)) = plt.subplots(2,3)  # Step axis, Episode axis, Time axis
fig.suptitle("Progress of 2048 AI over time")
fig.tight_layout()

sax.set_title("VS. Step")
if r64s: sax.plot(steps, r64s, label="64 rate")
if r128s: sax.plot(steps, r128s, label="128 rate")
if r256s: sax.plot(steps, r256s, label="256 rate")
if r512s: sax.plot(steps, r512s, label="512 rate")
if r1024s: sax.plot(steps, r1024s, label="1024 rate")
if r2048s: sax.plot(steps, r2048s, label="2048 rate")
if r4096s: sax.plot(steps, r4096s, label="4096 rate")
sax.vlines(barsteps, *sax.get_ylim(), color='red', label="Break between continues")
sax.set_xlabel("Step")
sax.set_ylabel("Rate (%)")
sax.grid(True)

eax.set_title("VS. Episode")
if r64s: eax.plot(episodes, r64s)
if r128s: eax.plot(episodes, r128s)
if r256s: eax.plot(episodes, r256s)
if r512s: eax.plot(episodes, r512s)
if r1024s: eax.plot(episodes, r1024s)
if r2048s: eax.plot(episodes, r2048s)
if r4096s: eax.plot(episodes, r4096s)
eax.vlines(barepisodes, *eax.get_ylim(), color='red')
eax.set_xlabel("Episode")
eax.set_ylabel("Rate (%)")
eax.grid(True)

tax.set_title("VS. Elapsed Time")
if r64s: tax.plot(times, r64s)
if r128s: tax.plot(times, r128s)
if r256s: tax.plot(times, r256s)
if r512s: tax.plot(times, r512s)
if r1024s: tax.plot(times, r1024s)
if r2048s: tax.plot(times, r2048s)
if r4096s: tax.plot(times, r4096s)
tax.vlines(bartimes, *tax.get_ylim(), color='red')
tax.set_xticklabels([int((dates.num2date(t).replace(tzinfo=None)-origin_dt).total_seconds())//3600 for t in tax.get_xticks()])
tax.set_xlabel("Elapsed Time (hrs)")
tax.set_ylabel("Rate (%)")
tax.grid(True)

fpsax.set_title("FPS VS. Time")
fpsax.plot(times[:-1], fps)
fpsax.set_yscale('log')
fpsax.vlines(bartimes, *fpsax.get_ylim(), color='red')
fpsax.set_xticklabels([int((dates.num2date(t).replace(tzinfo=None)-origin_dt).total_seconds())//3600 for t in fpsax.get_xticks()])
fpsax.set_xlabel("Elapsed Time (hrs)")
fpsax.set_ylabel("FPS")
fpsax.grid(True)

epsax.set_title("EPS VS. Time")
epsax.plot(times[:-1], eps)
epsax.set_yscale('log')
epsax.vlines(bartimes, *epsax.get_ylim(), color='red')
epsax.set_xticklabels([int((dates.num2date(t).replace(tzinfo=None)-origin_dt).total_seconds())//3600 for t in epsax.get_xticks()])
epsax.set_xlabel("Elapsed Time (hrs)")
epsax.set_ylabel("EPS")
epsax.grid(True)

# Show X on extra subplot
badax.plot([0,1],[0,1],c='grey')
badax.plot([1,0],[0,1],c='grey')

fig.legend()

plt.show()
