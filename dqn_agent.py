import torch
import torch.nn as nn
import torch.optim as optim
import collections
import random
import numpy as np
from datetime import datetime

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDQN(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, action_dim)
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        # Dueling aggregation
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
        self.q_values_history = []
        
    def log_episode(self, reward, length):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
    def log_training(self, loss, avg_q):
        self.loss_history.append(loss)
        self.q_values_history.append(avg_q)
        
    def get_stats(self, window=100):
        if len(self.episode_rewards) < window:
            return None
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'mean_length': np.mean(recent_lengths),
            'total_episodes': len(self.episode_rewards)
        }

class Agent:
    def __init__(self, state_dim=6, action_dim=2, hidden=128,
                 lr=3e-4, gamma=0.995, buffer_size=100000, batch_size=128):
        self.device = device
        self.dqn = DuelingDQN(state_dim, action_dim, hidden).to(self.device)
        self.target = DuelingDQN(state_dim, action_dim, hidden).to(self.device)
        self.target.load_state_dict(self.dqn.state_dict())
        self.target.eval()
        
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)
        self.loss_fn = nn.MSELoss()
        
        # Prioritized Experience Replay
        self.replay_buffer = collections.deque(maxlen=buffer_size)
        self.priorities = collections.deque(maxlen=buffer_size)
        self.alpha = 0.6  # prioritization strength
        self.beta = 0.4   # importance sampling
        self.beta_increment = 0.001
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.train_steps = 0
        
        # hyperparameters
        self.update_target_every = 1000
        self.n_step = 3  # n-step returns
        self.n_step_buffer = collections.deque(maxlen=self.n_step)
        
        # metrics
        self.metrics = MetricsTracker()

    def get_epsilon(self, episode, warmup=1000, min_eps=0.01):
        """Improved epsilon decay strategy"""
        if episode < warmup:
            return 1.0 - 0.9 * (episode / warmup)  # Fast initial decay
        else:
            # Exponential decay after warmup
            return max(min_eps, 0.1 * (0.995 ** (episode - warmup)))

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim-1)
        state_t = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.dqn(state_t)
            return int(q.argmax().item())

    def store(self, s, a, r, s2, done):
        # Store in n-step buffer first
        self.n_step_buffer.append((s, a, r, s2, done))
        
        if len(self.n_step_buffer) == self.n_step:
            # Create n-step transition
            n_step_reward = 0
            gamma_pow = 1
            
            for i in range(self.n_step):
                n_step_reward += gamma_pow * self.n_step_buffer[i][2]
                gamma_pow *= self.gamma
                if self.n_step_buffer[i][4]:  # if done
                    break
            
            # Store transition
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            sn, done_n = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
            
            self.replay_buffer.append((s0, a0, n_step_reward, sn, done_n))
            # Store with initial high priority for new experiences
            max_priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(max_priority)

    def train_step(self):
        if len(self.replay_buffer) < max(1000, self.batch_size):
            return
        
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Prioritized sampling
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs)
        weights = (len(self.replay_buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Sample batch
        batch = [self.replay_buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Current Q values
        q_vals = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target calculation
        with torch.no_grad():
            next_online_q = self.dqn(next_states)
            next_actions = next_online_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (self.gamma ** self.n_step) * next_target_q * (1 - dones)

        # Compute TD errors and loss
        td_errors = torch.abs(q_vals - target_q)
        loss = (weights * (q_vals - target_q) ** 2).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities
        td_errors_np = td_errors.detach().cpu().numpy()
        for idx, td_error in zip(indices, td_errors_np):
            self.priorities[idx] = td_error + 1e-6

        self.train_steps += 1
        
        # Update target network
        if self.train_steps % self.update_target_every == 0:
            self.target.load_state_dict(self.dqn.state_dict())
            
        # Update learning rate
        if self.train_steps % 100 == 0:
            self.scheduler.step()
            
        # Log metrics
        avg_q = q_vals.mean().item()
        self.metrics.log_training(loss.item(), avg_q)

    def save(self, path):
        """Save model with timestamp"""
        timestamp = datetime.now().strftime("%H%M")
        if not path.endswith('.pth'):
            path = f"{path}_{timestamp}.pth"
        else:
            path = path.replace('.pth', f'_{timestamp}.pth')
            
        torch.save({
            'model': self.dqn.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_steps': self.train_steps,
            'metrics': {
                'episode_rewards': self.metrics.episode_rewards,
                'episode_lengths': self.metrics.episode_lengths,
                'loss_history': self.metrics.loss_history[-1000:],  # Keep last 1000
                'q_values_history': self.metrics.q_values_history[-1000:]
            }
        }, path)
        return path

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        
        if 'model' in ckpt:
            self.dqn.load_state_dict(ckpt['model'])
            if 'target' in ckpt:
                self.target.load_state_dict(ckpt['target'])
            else:
                self.target.load_state_dict(self.dqn.state_dict())
        else:
            # Legacy support
            self.dqn.load_state_dict(ckpt)
            self.target.load_state_dict(self.dqn.state_dict())
            
        if 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception:
                pass
                
        if 'scheduler' in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            except Exception:
                pass
                
        if 'train_steps' in ckpt:
            self.train_steps = ckpt['train_steps']
            
        if 'metrics' in ckpt:
            metrics_data = ckpt['metrics']
            self.metrics.episode_rewards = metrics_data.get('episode_rewards', [])
            self.metrics.episode_lengths = metrics_data.get('episode_lengths', [])
            self.metrics.loss_history = metrics_data.get('loss_history', [])
            self.metrics.q_values_history = metrics_data.get('q_values_history', [])