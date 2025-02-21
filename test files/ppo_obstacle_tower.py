import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from datetime import datetime
from collections import deque
from torch.distributions import Categorical
import gym
from obstacle_tower_env import ObstacleTowerEnv

class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

class FrameStack:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        
    def reset(self, first_frame):
        for _ in range(self.n_frames):
            self.frames.append(first_frame)
        return self._get_observation()
    
    def step(self, frame):
        self.frames.append(frame)
        return self._get_observation()
    
    def _get_observation(self):
        return np.concatenate(list(self.frames), axis=0)

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.priorities = []

    def generate_batches(self, batch_size):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        
        states = np.array(self.states)
        actions = np.array(self.actions)
        probs = np.array(self.probs)
        vals = np.array(self.vals)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
    
        return states, actions, probs, vals, rewards, dones, batches    

    def store_memory(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        priority = abs(reward) + 0.1
        self.priorities.append(priority)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, action_dims):
        super(ActorNetwork, self).__init__()
        
        # Change first conv layer to accept 12 channels (4 frames x 3 channels)
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),  # Changed from 3 to 12
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc_input_dims = self._get_conv_output(input_dims)
        
        self.shared_layer = nn.Sequential(
            nn.Linear(self.fc_input_dims, 512),
            nn.ReLU()
        )
        
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, dim),
                nn.Softmax(dim=-1)
            ) for dim in action_dims
        ])

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self.conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, state):
        # Pass through convolutional layers
        conv_out = self.conv(state)
        # Flatten the output
        flat = conv_out.view(conv_out.size(0), -1)
        # Pass through shared fully connected layer
        shared_features = self.shared_layer(flat)
        
        # Get probabilities for each action dimension
        action_probs = [head(shared_features) for head in self.action_heads]
        
        return action_probs

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        
        # Change first conv layer to accept 12 channels (4 frames x 3 channels)
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),  # Changed from 3 to 12
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc_input_dims = self._get_conv_output(input_dims)
        
        self.critic = nn.Sequential(
            nn.Linear(self.fc_input_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self.conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, state):
        conv_out = self.conv(state)
        flat = conv_out.view(conv_out.size(0), -1)
        value = self.critic(flat)
        return value

class PPOAgent:
    def __init__(self, input_dims, action_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.steps = 0
        self.epsilon = 1.0  # For exploration

        # Main networks
        self.actor = ActorNetwork(input_dims, action_dims).to(self.device)
        self.critic = CriticNetwork(input_dims).to(self.device)
        
        # Target networks
        self.target_actor = ActorNetwork(input_dims, action_dims).to(self.device)
        self.target_critic = CriticNetwork(input_dims).to(self.device)
        self.update_target_interval = 1000

        self.optimizer = optim.Adam(list(self.actor.parameters()) + 
                                  list(self.critic.parameters()), lr=alpha)
        
        self.memory = PPOMemory()
        self.experience_buffer = ExperienceBuffer()
        self.batch_size = batch_size
        self.action_dims = action_dims

    def save(self, filename):
        """Save the model's state dictionaries"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load the model's state dictionaries"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filename}")

    def store_transition(self, state, action, probs, val, reward, done):
        """
        Store a transition in memory
        state: the observation
        action: array of actions for each dimension
        probs: list of probabilities for each chosen action
        val: value estimate
        reward: reward received
        done: whether episode ended
        """
        # Calculate total probability as product of individual action probabilities
        total_prob = np.prod(probs)
        self.memory.store_memory(state, action, total_prob, val, reward, done)

    def choose_action(self, observation):
        # Convert numpy array to torch tensor
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).to(self.device)
        
        # Epsilon-greedy exploration
        self.epsilon = max(0.01, 1.0 - (self.steps / 50000))
        if random.random() < self.epsilon:
            random_actions = np.array([random.randint(0, dim-1) for dim in self.action_dims])
            # Instead of None, use a uniform probability distribution
            random_probs = np.ones(len(self.action_dims)) / len(self.action_dims)
            return (random_actions, 
                    random_probs,  # Return actual probabilities instead of None
                    self.critic(observation.unsqueeze(0)).item())
        
        state = observation.unsqueeze(0)
        action_distributions = self.actor(state)
        
        actions = []
        probs = []
        for dist in action_distributions:
            action = Categorical(dist).sample()
            prob = dist[0][action]
            actions.append(action.item())
            probs.append(prob.item())
        
        self.steps += 1
        
        if self.steps % self.update_target_interval == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
            
        return np.array(actions), np.array(probs), self.critic(state).item()

    def learn(self):
        advantages_list = []  # Store all advantages
        states_list = []      # Store all states
        actions_list = []     # Store all actions
        if len(self.experience_buffer.buffer) >= self.batch_size * 4:
            experiences = random.sample(self.experience_buffer.buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # Calculate advantages for experience replay
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            advantages = rewards + self.gamma * next_values * (1 - dones) - values
            
            # Store for combined learning
            advantages_list.append(advantages)
            states_list.append(states)

        # Then process regular PPO memory
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches(self.batch_size)

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calculate GAE advantages
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            # Convert to tensors
            advantage = torch.tensor(advantage).to(self.device)
            values = torch.tensor(values).to(self.device)
            
            # Add to combined advantages
            advantages_list.append(advantage)
            states_list.append(torch.tensor(state_arr, dtype=torch.float).to(self.device))
            actions_list.append(torch.tensor(action_arr, dtype=torch.long).to(self.device))

            # Combine all advantages and data
            combined_advantages = torch.cat(advantages_list)
            combined_states = torch.cat(states_list)
            combined_actions = torch.cat(actions_list)

            # Use combined data for policy update
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float).to(self.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.long).to(self.device)

                dist_list = self.actor(combined_states[batch])
                critic_value = self.critic(combined_states[batch])
                critic_value = torch.squeeze(critic_value)

                new_log_probs = []
                for dim_idx, (dist, action_dim) in enumerate(zip(dist_list, actions.T)):
                    action_dim_probs = dist.gather(1, action_dim.unsqueeze(1)).squeeze()
                    new_log_probs.append(torch.log(action_dim_probs + 1e-10))
                
                new_log_probs = torch.stack(new_log_probs, dim=1).sum(dim=1)
                old_log_probs = torch.log(old_probs + 1e-10)

                prob_ratio = torch.exp(new_log_probs - old_log_probs)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            self.memory.clear_memory()

def get_curriculum_floor(episode):
    if episode < 50:
        return 0  # Start with floor 0
    elif episode < 100:
        return random.randint(0, 1)  # Then introduce floor 1
    else:
        return -1  # Let it try any floor
    
def get_difficulty(episode):
    if episode < 50:
        return 0
    elif episode < 100:
        return min(1, episode / 100)
    return 1
    
def shaped_reward(reward, info, prev_position, current_position, episode):
    total_reward = reward
    
    try:
        movement_delta = np.linalg.norm(current_position - prev_position)
        
        # Increase movement rewards
        if movement_delta > 0:
            total_reward += 0.05  # Increased from 0.01
            
            # Larger reward for forward progress
            if current_position[0] > prev_position[0]:
                total_reward += 0.1  # Increased from 0.02
        
        # Less harsh penalty
        if movement_delta < 0.001:
            total_reward -= 0.0005  # Reduced penalty
            
    except:
        return reward
        
    return total_reward

def preprocess_image(observation):
    # print("Observation type:", type(observation))
    # print("Observation shape:", observation.shape)
    
    # Check if the observation is in the right format
    if len(observation.shape) == 3 and observation.shape[-1] == 3:  # HWC format
        return np.transpose(observation, (2, 0, 1))  # Convert to CHW
    elif len(observation.shape) == 3 and observation.shape[0] == 3:  # Already in CHW
        return observation
    else:
        print("Unexpected observation shape:", observation.shape)
        # Handle the unexpected shape
        return observation

def main():
    # Set up logging to file
    log_file = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    env = ObstacleTowerEnv('./ObstacleTower/ObstacleTower.exe', worker_id=1, retro=False, 
                          realtime_mode=False)
    frame_stack = FrameStack(4)
    
    input_dims = (12, 84, 84)
    action_dims = env.action_space.nvec
    
    agent = PPOAgent(input_dims=input_dims, 
                    action_dims=action_dims,
                    batch_size=64,
                    alpha=0.0001)
    
    best_score = float('-inf')
    scores = []
    total_steps = 0
    
    # Log initial setup
    with open(log_file, 'w') as f:
        f.write(f"Training started at: {datetime.now()}\n")
        f.write(f"Action dimensions: {action_dims}\n\n")
    
    for episode in range(300):
        difficulty = get_difficulty(episode)
        obs_tuple = env.reset()
        observation = preprocess_image(obs_tuple[0])
        stacked_obs = frame_stack.reset(observation)
        
        done = False
        score = 0
        raw_score = 0
        shaped_score = 0
        steps = 0
        prev_position = np.zeros(3)
        
        while not done:
            action, log_prob, val = agent.choose_action(stacked_obs)
            next_obs_tuple, reward, done, info = env.step(action)
            next_observation = preprocess_image(next_obs_tuple[0])
            
            current_position = np.array(info.get('position', np.zeros(3)))
            shaped_rew = shaped_reward(reward, info, prev_position, current_position, episode)
            
            agent.experience_buffer.add(stacked_obs, action, shaped_rew, 
                                     frame_stack.step(next_observation), done)
            
            agent.store_transition(stacked_obs, action, log_prob, val, shaped_rew, done)
            
            if len(agent.memory.states) >= agent.batch_size:
                agent.learn()
            
            stacked_obs = frame_stack.step(next_observation)
            raw_score += reward
            shaped_score += shaped_rew
            score = raw_score
            prev_position = current_position
            steps += 1
            total_steps += 1
        
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        
        # Save best model
        if score > best_score:
            best_score = score
            agent.save(f'best_model_score_{best_score:.2f}.pt')
            best_msg = f'New best score! {best_score:.2f} - Model saved'
            print(best_msg)
            with open(log_file, 'a') as f:
                f.write(best_msg + '\n')
        
        # Log every 50 episodes or if it's a particularly good episode
        if episode % 50 == 0 or score > 0:
            log_msg = (f'Episode: {episode}, '
                      f'Raw Score: {raw_score:.2f}, '
                      f'Shaped Score: {shaped_score:.3f}, '
                      f'Avg Score: {avg_score:.2f}, '
                      f'Best Score: {best_score:.2f}, '
                      f'Steps: {steps}, '
                      f'Epsilon: {agent.epsilon:.3f}, '
                      f'Total Steps: {total_steps}\n')
            
            print(log_msg)  # Print to console
            with open(log_file, 'a') as f:
                f.write(log_msg)  # Write to file
        
        # Save checkpoint every 50 episodes
        if episode % 50 == 0:
            agent.save(f'checkpoint_episode_{episode}.pt')
            
    # Log final statistics
    final_msg = f"\nTraining completed at: {datetime.now()}\n"
    final_msg += f"Final Best Score: {best_score:.2f}\n"
    final_msg += f"Total Steps: {total_steps}\n"
    
    with open(log_file, 'a') as f:
        f.write(final_msg)
    print(final_msg)
    
    env.close()

if __name__ == '__main__':
    main()