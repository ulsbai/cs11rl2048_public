# cs11rl2048

cs11rl2048 is my Computer Science 11 project in 2025&ndash;26. My project is about training a reinforcement learning agent to play the game of 2048.

## Usage

### Installation

1. In the `main` branch of the repository on GitHub, click on Code > Download ZIP.
2. Unzip the downloaded zip file and go to it in your terminal
```sh
cd Downloads
unzip cs11rl2048_public-main.zip
cd cs11rl2048_public-main
```
3. Make a virtual environment (optional)
```sh
python3 -m venv venv
source venv/bin/activate
```
4. Install the requirements
```sh
pip3 install -r requirements.txt
```
In my experience, PyTorch (one of the required packages) may not install on MacOS until pip is upgraded:
```
pip3 install -U pip
```

### Playing 2048 yourself

To play my implementation of 2048, run
```sh
python3 src/_2048.py
```

### Training a deep Q-learning model

To train a deep Q-learning model with a convolutional neural network and my most successful hyperparameters, run
```sh
python3 -u src/train_2048_dqn_model.py |& tee -i train_log.txt
```
We are saving a copy of the output to `train_log.txt` so that later we can [plot the learning curve](#plotting-the-learning-curve).

The program will print information about its progress every 100 games it plays, such as
```
Step: 81845
Episode: 100
Elapsed time: 0:02:00.571489
Avg episode length: 818.450
Avg step reward: 16.710
Avg episode reward: 13676.600
Avg loss: 7665.854
Epsilon: 0.009
64 rate: 100.0%
128 rate: 99.0%
256 rate: 99.0%
512 rate: 95.0%
1024 rate: 73.0%
2048 rate: 8.0%
4096 rate: 0.0%
```
With these hyperparameters, you will likely need to wait several hours to see improvement. After training for 5 hours on a 3.8GHz 8-core Intel Core i7 iMac with 64 GB of RAM, it reached 256 approximately 45% of the time, compared to a 6% 256 rate when starting training.

Every 1000 games, the program will save its models and transitions and print a message such as
```
Saving state...
Target net saved to file pth_models/target_net_6535060316876111615.pth
Policy net saved to file pth_models/policy_net_6535060316876111615.pth
Transition buffer saved to file transition_buffers/transitions_6535060316876111615.pickle
Hyperparameters saved to file hyperparameters/hyperparameters_6535060316876111615.txt
Finished saving state.
```
The program generates a random number to use in the filename to ensure that a new training run does not overwrite any model file from an older training run.

Press `Ctrl+C` to stop the program and it will automatically save its models before quitting.

You can change the hyperparameters by editing the file `src/train_2048_dqn_model.py`. The hyperparameters are constants at the beggining of the `if __name__ == '__main__':` section.

### Testing a trained deep Q-learning model

To test the trained model, use a command of the form
```
python3 src/test_2048_dqn_model.py -f=<model_file> [--render] [-n=<num_episodes>]
```
Replace `<model_file>` with the filename of the target network copy-pasted from the training run's saving message; for example, `pth_models/target_net_6535060316876111615.pth`. The model will play `<num_episodes>` (default 100) games and print how well it did at the end. With `--render`, it will show you the games it plays on the screen, and the gameplay is deliberately slowed down to 1 second delay per move to allow a human to look at it.

### Plotting the learning curve

To plot the progress of the training over time, run the command
```sh
python3 plot_progress.py -f=train_log.txt
```
where `train_log.txt` is the file containing the output of the training run. The program parses the stats that the training program has printed and graphs them against the step (move) count, episode (game) count, and runtime in seperate subplots. It also plots the number of steps/frames per second (fps) and episodes per second (eps) that the model plays during training. You can also run this command before the training is finished to see the model's progress so far.

### Interpretability

To run interpretability on a model, use a command of the form
```sh
python3 src/interpret_2048_dqn_model.py -f=<model_file> --game-obs|--rand-obs --symins|--lime
```
Replace `<model_file>` with filename of the saved target network. Use `--symins` to show symmetry insanities, and `--lime` to run Local Interpretable Model-agnostic Explanations (LIME). After running LIME, the program will show two graphs, for attributes versus tile value and versus tile position. Use `--game-obs` and `--rand-obs` for real-game and random input observations, respectively. See the project report document for a more detailed description of these concepts.
