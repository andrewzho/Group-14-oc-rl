import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from obstacle_tower_env import ObstacleTowerEnv

class MetaController(nn.Module):
    def __init__(self):
        super().__init__()
        # Input will be stacked frames (4 frames x 3 channels = 12)
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        
        # Calculate fc input dimension
        self.fc_input_dim = self._get_conv_output((12, 84, 84))
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # 3 subtasks: explore, get key, reach door
        )

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.zeros(bs, *shape)
        output = self.cnn(input)
        return int(np.prod(output.shape))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SubController(nn.Module):
    def __init__(self, action_dims):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        
        self.fc_input_dim = self._get_conv_output((12, 84, 84))
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU()
        )
        
        # Create separate output layers for each action dimension
        self.action_heads = nn.ModuleList([
            nn.Linear(512, dim) for dim in action_dims
        ])

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.zeros(bs, *shape)
        output = self.cnn(input)
        return int(np.prod(output.shape))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return [F.softmax(head(x), dim=-1) for head in self.action_heads]

class HRLAgent:
    def __init__(self, state_dim, action_dims, gamma=0.99, alpha=0.0003):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.meta_controller = MetaController().to(self.device)
        self.sub_controllers = [SubController(action_dims).to(self.device) for _ in range(3)]
        self.action_dims = action_dims
        
        # Optimizers
        self.meta_optimizer = optim.Adam(self.meta_controller.parameters(), lr=alpha)
        self.sub_optimizers = [optim.Adam(controller.parameters(), lr=alpha) 
                             for controller in self.sub_controllers]
        
        # Experience replay buffers
        self.meta_buffer = deque(maxlen=100000)
        self.sub_buffers = [deque(maxlen=100000) for _ in range(3)]
        
        # Parameters
        self.gamma = gamma
        self.batch_size = 32
        self.subtask_duration = 50  # Steps before switching subtasks
        
        # Current state
        self.current_subtask = None
        self.subtask_steps = 0
        self.frame_stack = deque(maxlen=4)
        
        # Initialize epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def preprocess_state(self, state):
        """Convert state tuple to properly formatted tensor"""
        if isinstance(state, tuple):
            image = state[0]  # Extract image from tuple
        else:
            image = state
            
        # Ensure image is float and normalized
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        # Add to frame stack
        if len(self.frame_stack) < 4:
            for _ in range(4):
                self.frame_stack.append(image)
        else:
            self.frame_stack.append(image)
            
        # Stack frames along channel dimension
        stacked_frames = np.concatenate(list(self.frame_stack), axis=2)
        
        # Convert to tensor and reshape to (C, H, W)
        state_tensor = torch.FloatTensor(stacked_frames).permute(2, 0, 1)
        return state_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

    def choose_action(self, state):
        state_tensor = self.preprocess_state(state)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            actions = [random.randint(0, dim-1) for dim in self.action_dims]
            return np.array(actions)
            
        with torch.no_grad():
            if self.current_subtask is None or self.subtask_steps >= self.subtask_duration:
                # Choose new subtask
                meta_output = self.meta_controller(state_tensor)
                self.current_subtask = torch.argmax(meta_output).item()
                self.subtask_steps = 0
                
            # Get action from current subtask controller
            action_probs = self.sub_controllers[self.current_subtask](state_tensor)
            actions = [torch.argmax(probs[0]).item() for probs in action_probs]
            
        self.subtask_steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return np.array(actions)

    def store_transition(self, state, action, reward, next_state, done):
        # Initialize subtask if None
        if self.current_subtask is None:
            state_tensor = self.preprocess_state(state)
            with torch.no_grad():
                meta_output = self.meta_controller(state_tensor)
                self.current_subtask = torch.argmax(meta_output).item()
                self.subtask_steps = 0

        # Store in meta controller buffer
        self.meta_buffer.append((
            self.preprocess_state(state).cpu().numpy(),
            self.current_subtask,
            reward,
            self.preprocess_state(next_state).cpu().numpy(),
            done
        ))
        
        # Store in current subtask buffer
        self.sub_buffers[self.current_subtask].append((
            self.preprocess_state(state).cpu().numpy(),
            action,
            reward,
            self.preprocess_state(next_state).cpu().numpy(),
            done
        ))

    def learn(self):
        if len(self.meta_buffer) < self.batch_size:
            return
            
        # Sample from meta controller buffer
        meta_batch = random.sample(self.meta_buffer, self.batch_size)
        states, subtasks, rewards, next_states, dones = zip(*meta_batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).squeeze(1).to(self.device)
        subtasks = torch.LongTensor(subtasks).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).squeeze(1).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update meta controller
        self.meta_optimizer.zero_grad()
        current_q = self.meta_controller(states)
        next_q = self.meta_controller(next_states).detach()
        
        target_q = rewards + (1 - dones) * self.gamma * torch.max(next_q, dim=1)[0]
        meta_loss = F.mse_loss(current_q.gather(1, subtasks.unsqueeze(1)).squeeze(), target_q)
        
        meta_loss.backward()
        self.meta_optimizer.step()
        
        # Update sub controllers
        for i, buffer in enumerate(self.sub_buffers):
            if len(buffer) < self.batch_size:
                continue
                
            batch = random.sample(buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(np.array(states)).squeeze(1).to(self.device)
            actions = torch.LongTensor(np.array(actions)).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).squeeze(1).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            self.sub_optimizers[i].zero_grad()
            
            current_probs = self.sub_controllers[i](states)
            next_probs = self.sub_controllers[i](next_states)
            
            sub_loss = 0
            for dim in range(len(self.action_dims)):
                action_probs = current_probs[dim]
                next_action_probs = next_probs[dim].detach()
                
                target = rewards + (1 - dones) * self.gamma * torch.max(next_action_probs, dim=1)[0]
                sub_loss += F.mse_loss(
                    action_probs.gather(1, actions[:, dim].unsqueeze(1)).squeeze(),
                    target
                )
            
            sub_loss.backward()
            self.sub_optimizers[i].step()

    def reset(self):
        """Reset agent's episode-specific variables"""
        self.current_subtask = None
        self.subtask_steps = 0
        self.frame_stack.clear()

def main():
    try:
        # Initialize environment
        env = ObstacleTowerEnv(
            worker_id=1,
            retro=False,
            realtime_mode=False
        )
    
        # Initialize agent
        state_dim = (12, 84, 84)  # 4 stacked frames x 3 channels
        action_dims = env.action_space.nvec  # Get action dimensions from environment
        agent = HRLAgent(state_dim, action_dims)
        
        num_episodes = 1000
        max_steps = 1000
        
        for episode in range(num_episodes):
            state = env.reset()
            agent.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                # Only learn if we have enough samples
                if len(agent.meta_buffer) >= agent.batch_size:
                    agent.learn()
                
                state = next_state
                episode_reward += reward

                # Print progress
                if step % 100 == 0:
                    print(f"Episode {episode}, Step {step}, Current Reward: {episode_reward}")
                
                if done:
                    break
            
            print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")
            
            # Save model periodically
            if episode % 100 == 0:
                torch.save({
                    'meta_controller': agent.meta_controller.state_dict(),
                    'sub_controllers': [c.state_dict() for c in agent.sub_controllers],
                    'episode': episode,
                    'epsilon': agent.epsilon
                }, f'hrl_checkpoint_{episode}.pt')
    
    finally:
        env.close()
        print("Environment closed.")

if __name__ == '__main__':
    main()