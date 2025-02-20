import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from obstacle_tower_env import ObstacleTowerEnv
from datetime import datetime
import time

# Actor network (policy)
class Actor(nn.Module):
    def __init__(self, input_dims, action_dims):
        super(Actor, self).__init__()
        
        # Store the action dimensions
        self.action_dims = action_dims
        self.total_actions = sum(action_dims)  # Total number of actions across all dimensions
        
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc_input_dims = self._get_conv_output((12, 84, 84))
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dims, 512),
            nn.ReLU()
        )
        
        # Create separate outputs for each action dimension
        self.action_heads = nn.ModuleList([
            nn.Linear(512, dim) for dim in action_dims
        ])

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output = self.conv(input)
        return int(np.prod(output.shape))

    def forward(self, state):
        x = self.conv(state)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # Get logits for each action dimension
        action_logits = [head(x) for head in self.action_heads]
        
        return action_logits


# Critic network (Q-value)
class Critic(nn.Module):
    def __init__(self, input_dims):
        super(Critic, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc_input_dims = self._get_conv_output((12, 84, 84))
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 256)  # Changed to match expected output size
        )

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output = self.conv(input)
        return int(np.prod(output.shape))

    def forward(self, state):
        x = self.conv(state)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        q_values = self.fc(x)
        return q_values

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, input_dims, action_dims, alpha=0.001, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.action_dims = action_dims

        # Initialize networks
        self.actor = Actor(input_dims, action_dims).to(self.device)
        self.critic1 = Critic(input_dims).to(self.device)
        self.critic2 = Critic(input_dims).to(self.device)
        self.target_critic1 = Critic(input_dims).to(self.device)
        self.target_critic2 = Critic(input_dims).to(self.device)

        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=alpha)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=alpha)

        # Initialize memory
        self.memory = ReplayBuffer(100000)
        
        # Temperature parameter
        self.alpha = alpha
        self.target_entropy = -np.prod(action_dims).item()  # Heuristic value
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha)

        self.frame_stack = deque(maxlen=4)
    def stack_frames(self, frame):
        """Stack 4 frames together"""
        if len(self.frame_stack) == 0:  # Initialize stack if empty
            for _ in range(4):
                self.frame_stack.append(frame)
        else:
            self.frame_stack.append(frame)
        
        # Stack frames along channel dimension
        stacked_frames = np.concatenate(list(self.frame_stack), axis=2)
        return stacked_frames

    def choose_action(self, state):
        with torch.no_grad():
            # Extract the image component and stack frames
            image = state[0]
            stacked_frames = self.stack_frames(image)
            
            # Convert to tensor and permute dimensions from HWC to CHW format
            image_tensor = torch.FloatTensor(stacked_frames).unsqueeze(0)
            image_tensor = image_tensor.permute(0, 3, 1, 2)
            image_tensor = image_tensor.to(self.device)
            
            # Get action logits from actor network
            action_logits = self.actor(image_tensor)
            
            # Sample actions for each dimension
            actions = []
            for logits in action_logits:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
                actions.append(action)
            
            # Convert to numpy array with correct shape (1, 4)
            actions = np.array(actions).reshape(1, -1)
            
            return actions

    def reset(self):
        """Clear the frame stack when environment resets"""
        self.frame_stack.clear()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def update_target_networks(self):
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self, batch_size=256):
        if len(self.memory) < batch_size:
            return

        # Move to GPU in batch for better performance
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_images = torch.FloatTensor(np.array([self.stack_frames(s[0]) for s in states]))
        next_state_images = torch.FloatTensor(np.array([self.stack_frames(s[0]) for s in next_states]))

        # Move all tensors to device at once
        state_tensor = state_images.permute(0, 3, 1, 2).to(self.device)
        next_state_tensor = next_state_images.permute(0, 3, 1, 2).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).to(self.device)
        done_tensor = torch.FloatTensor(dones).to(self.device)
        action_tensor = torch.FloatTensor(np.array(actions)).to(self.device)

        # Update critics
        with torch.no_grad():
            next_action_logits = self.actor(next_state_tensor)
            next_actions = []
            next_log_probs = []
            
            for logits in next_action_logits:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                next_actions.append(action)
                next_log_probs.append(log_prob)

            next_q1 = self.target_critic1(next_state_tensor)
            next_q2 = self.target_critic2(next_state_tensor)
            next_q = torch.min(next_q1, next_q2)
            target_q = reward_tensor.unsqueeze(1) + (1 - done_tensor.unsqueeze(1)) * self.gamma * next_q

        # Update critics
        current_q1 = self.critic1(state_tensor)
        current_q2 = self.critic2(state_tensor)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        action_logits = self.actor(state_tensor)
        actor_loss = 0
        for logits in action_logits:
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            q1 = self.critic1(state_tensor)
            q2 = self.critic2(state_tensor)
            q = torch.min(q1, q2)
            actor_loss += -(q - self.alpha * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature parameter
        alpha_loss = 0
        for log_prob in next_log_probs:
            alpha_loss += -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Update target networks
        self.update_target_networks()

def main():
    env = ObstacleTowerEnv('./ObstacleTower/ObstacleTower.exe', worker_id=1, retro=False, 
                          realtime_mode=False)
    
    input_dims = (12, 84, 84)
    action_dims = list(env.action_space.nvec)
    agent = SACAgent(input_dims=input_dims, action_dims=action_dims)
    
    best_score = float('-inf')
    scores = []
    
    print("Starting training...")
    
    for episode in range(300):
        start_time = time.time()  # Track episode time
        obs = env.reset()
        agent.reset()
        done = False
        score = 0
        steps = 0
        
        while not done and steps < 1000:  # Add step limit to prevent infinite loops
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            agent.store_transition(obs, action, reward, next_obs, done)
            
            # Only learn every 4 steps to improve performance
            if steps % 4 == 0:
                agent.learn()
            
            obs = next_obs
            score += reward
            steps += 1
            
            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"Episode {episode} - Step {steps}")
        
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        episode_time = time.time() - start_time
        
        print(f'Episode: {episode}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, '
              f'Steps: {steps}, Time: {episode_time:.2f}s')
        
        if score > best_score:
            best_score = score
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic1_state_dict': agent.critic1.state_dict(),
                'critic2_state_dict': agent.critic2.state_dict(),
            }, f'sac_best_model_{best_score:.2f}.pt')
            print(f'New best score! {best_score:.2f} - Model saved')
        
        # Save checkpoint every 10 episodes
        if episode % 10 == 0:
            torch.save({
                'episode': episode,
                'actor_state_dict': agent.actor.state_dict(),
                'critic1_state_dict': agent.critic1.state_dict(),
                'critic2_state_dict': agent.critic2.state_dict(),
                'scores': scores,
            }, 'sac_checkpoint.pt')
    
    env.close()

if __name__ == '__main__':
    main()