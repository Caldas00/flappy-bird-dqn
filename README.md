# Flappy DQN

A small, self-made Flappy Bird game written in [pygame](https://www.pygame.org/), wrapped as a Gym-style environment, plus a Deep Q-Network agent that learns to play it from scratch.

The whole project is three Python files: a game/environment, an agent, and a training script. There are no pretrained models in this repo — you train your own.

## What's in the repo

| File | What it does |
|---|---|
| `flappy_game.py` | The Flappy Bird game and its RL environment (`FlappyBirdEnv`). Physics, pipes, collisions, rewards, rendering. |
| `dqn_agent.py` | The learning side: Dueling Q-network, replay buffer, training step, checkpoint save/load, metrics. |
| `main.py` | Interactive training script: menu, training loop, auto-saves, statistics. |
| `requirements.txt` | Python dependencies. |

## How it works

### The environment (`flappy_game.py`)

`FlappyBirdEnv` follows the usual Gym pattern, so any RL algorithm can be plugged into it:

```python
env = FlappyBirdEnv(render=False)   # render=True opens a pygame window
state = env.reset()                 # -> np.array of 6 floats
state, reward, done = env.step(a)   # a = 0 (do nothing) or 1 (flap)
```

It is Gym-*style* (`reset()` / `step()`), but it is not a `gymnasium.Env` subclass, and `step` returns `(state, reward, done)` instead of Gymnasium's 5-tuple. Wrapping it in a real `gymnasium.Env` would be a small change.

**Game:** a 400×600 window with a bird at a fixed x position and one pipe pair at a time. Gravity pulls the bird down every frame; a flap sets its vertical velocity to a fixed upward value. The bird has a radius, so collisions with the pipes, the ceiling and the ground are checked against its body, not a single point. An episode ends on any collision.

**State (6 values, all normalised):**

1. bird height
2. bird vertical velocity (clipped to [-1, 1])
3. horizontal distance to the next pipe
4. vertical distance to the top edge of the gap
5. vertical distance to the bottom edge of the gap
6. vertical distance to the centre of the gap

**Rewards:**

| Event | Reward |
|---|---|
| Surviving a frame | +0.1 |
| Passing a pipe | +10 |
| Dying | −10 |
| Being inside the gap | up to +0.01 extra, larger the closer to the centre (a small shaping signal) |

### The agent (`dqn_agent.py`)

The agent is a DQN with several standard improvements on top of the vanilla algorithm:

- **Dueling network:** 6 inputs → three hidden layers of 128 units (ReLU) → separate *value* and *advantage* heads, combined into 2 Q-values (do nothing / flap).
- **Double DQN:** the online network picks the next action and the target network evaluates it, which reduces Q-value overestimation.
- **Prioritised experience replay:** transitions with a larger TD error are sampled more often (α = 0.6), with importance-sampling weights that anneal from β = 0.4 to 1.0.
- **3-step returns:** each stored transition is the discounted sum of the next three rewards.
- **Target network:** hard-copied from the online network every 1000 training steps.
- **Optimisation:** Adam (lr 3e-4, decayed with a step scheduler), γ = 0.995, batch size 128, replay buffer of 100k transitions, gradient clipping at 1.0.

Learning starts once the buffer holds 1000 transitions, and one gradient step is taken per environment step.

### Exploration

ε-greedy, decided per episode:

- Episodes 0–1000: ε decays linearly from 1.0 to 0.1.
- After that: ε decays exponentially (×0.995 per episode) down to a floor of 0.01.

## Setup

Tested with Python 3.13. (pygame's font module did not work for me on Python 3.14, so use 3.13 if you hit that.)

```bash
git clone git@github.com:Caldas00/flappy-bird-dqn.git
cd flappy-bird-dqn

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python main.py
```

You will be asked two things:

1. **New network or load an existing one?** Loading lists the `.pth` files found in `models/` and resumes from that checkpoint, including the optimiser state, step counter and episode history.
2. **Training mode:**
   - *Visual* — renders every episode so you can watch the bird learn (slower).
   - *Turbo* — no rendering, much faster.
   - *Hybrid* — visual for the first 1000 episodes, then switches to turbo.

Training runs for 5000 episodes by default (`max_episodes` in `main.py`). If you load a checkpoint that has already reached that, it asks how many extra episodes to run.

While a window is open:

- `S` — stop training
- `Space` — save a checkpoint immediately
- closing the window also stops training

### What gets saved

Everything goes in `models/` (created automatically, and ignored by git):

- **Auto-save every 300 episodes:** a timestamped `.pth` checkpoint plus a `_stats.txt` summary (mean/max reward and episode length over the last 1000 episodes, training steps, recent loss).
- **Final save:** at the end it asks for a filename (Enter accepts an automatic one).

Each checkpoint stores the online and target networks, optimiser, scheduler, step count and the reward/length history, so training can be resumed exactly where it stopped.

## Notes

- Training with a GPU is picked up automatically if CUDA is available, but this network is small and runs fine on CPU.
