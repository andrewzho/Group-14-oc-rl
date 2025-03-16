import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import datetime
import traceback
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main src directory
from src.model import PPONetwork, RecurrentPPONetwork
from src.ppo import PPO
from src.utils import ActionFlattener, save_checkpoint, load_checkpoint, MetricsTracker
from src.create_env import create_obstacle_tower_env

# Import from multi_agent
from multi_agent.vec_env import ObstacleTowerVecEnv
from multi_agent.env_wrapper import make_vec_envs

# Set up logging
import logging
logger = logging.getLogger("multi_agent")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def create_log_dir(base_dir, run_name=None):
    """Create and return a log directory with the current timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        log_dir = os.path.join(base_dir, f"{run_name}_{timestamp}")
    else:
        log_dir = os.path.join(base_dir, f"multi_agent_run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def initialize_tensorboard(log_dir):
    """Initialize and return a TensorBoard writer."""
    return SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

def preprocess_obs_batch(obs_batch, device):
    """
    Preprocess a batch of observations for the neural network.
    
    Args:
        obs_batch: Batch of observations from environments
        device: Device to place the tensor on
        
    Returns:
        Tensor of preprocessed observations
    """
    try:
        # Handle already tensor data
        if isinstance(obs_batch, torch.Tensor):
            return obs_batch.to(device)
            
        # Handle None or empty input
        if obs_batch is None or len(obs_batch) == 0:
            raise ValueError("Empty observation batch received")
            
        # Handle list of numpy arrays or list of other types
        if isinstance(obs_batch, list):
            # Check for None values in the list
            for i, obs in enumerate(obs_batch):
                if obs is None:
                    # Replace None with zeros
                    shape = (12, 84, 84)  # Default observation shape (4 frames x 3 channels x 84 x 84)
                    obs_batch[i] = np.zeros(shape, dtype=np.float32)
            
            # Convert to numpy if not already
            obs_batch = np.array(obs_batch, dtype=np.float32)
        
        # Now convert numpy array to tensor
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)
        
        # Move to device
        return obs_tensor.to(device)
    except Exception as e:
        # Log and handle error
        print(f"Error in preprocess_obs_batch: {e}")
        # Return a zero tensor as fallback
        batch_size = len(obs_batch) if isinstance(obs_batch, list) else 1
        shape = (batch_size, 12, 84, 84)  # Default observation shape
        return torch.zeros(shape, dtype=torch.float32, device=device)

def evaluate_multi_agent(
    model, 
    env_path, 
    num_envs=4, 
    num_episodes=20,
    seed=0, 
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate the model across multiple parallel environments.
    
    Args:
        model: The policy model to evaluate
        env_path: Path to the Obstacle Tower executable
        num_envs: Number of parallel environments to use
        num_episodes: Total number of episodes to evaluate
        seed: Random seed
        device: Device to run evaluation on
        
    Returns:
        mean_reward, max_floor: Evaluation metrics
    """
    logger.info(f"Starting evaluation with {num_envs} environments...")
    
    # Create vectorized environment for evaluation
    eval_envs = make_vec_envs(
        executable_path=env_path,
        num_envs=num_envs,
        seed=seed + 1000,  # Use a different seed from training
        timeout=300,
        realtime_mode=False
    )
    
    # Create ActionFlattener for MultiDiscrete action space
    if hasattr(eval_envs.action_space, 'n'):
        action_dim = eval_envs.action_space.n
        action_flattener = None
    else:
        action_flattener = ActionFlattener(eval_envs.action_space.nvec)
        action_dim = action_flattener.action_space.n
        
    # Check if the model is recurrent
    is_recurrent = hasattr(model, 'use_lstm') and model.use_lstm
    
    # Tracking variables
    episode_rewards = []
    episode_floors = []
    episodes_completed = 0
    max_floors = [0] * num_envs
    current_rewards = [0] * num_envs
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize LSTM states if needed
    lstm_states = None
    if is_recurrent:
        lstm_states = model.init_lstm_state(batch_size=num_envs, device=device)
    
    # Reset all environments
    obs = eval_envs.reset()
    obs_tensor = preprocess_obs_batch(obs, device)
    
    # Initialize episode_starts for LSTM
    episode_starts = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    active_envs = [True] * num_envs
    active_count = num_envs
    
    while episodes_completed < num_episodes:
        with torch.no_grad():
            # Forward pass through the model
            if is_recurrent:
                policy_logits, values, next_lstm_states = model(obs_tensor, lstm_states, episode_starts)
                lstm_states = next_lstm_states
                # Reset episode_starts after use
                episode_starts = torch.zeros(num_envs, dtype=torch.bool, device=device)
            else:
                policy_logits, values = model(obs_tensor)
            
            # Select deterministic actions for evaluation
            actions = torch.argmax(policy_logits, dim=1).cpu().numpy()
            
            # Convert actions if needed
            if action_flattener:
                actions = [action_flattener.lookup_action(a) for a in actions]
            
            # Step environments
            next_obs, rewards, dones, infos = eval_envs.step(actions)
            
            # Update episode_starts for environments that finished episodes
            if is_recurrent:
                episode_starts = torch.tensor(dones, dtype=torch.bool, device=device)
            
            # Update tracking variables
            for i in range(num_envs):
                if active_envs[i]:
                    current_rewards[i] += rewards[i]
                    
                    # Track highest floor
                    if 'current_floor' in infos[i]:
                        max_floors[i] = max(max_floors[i], infos[i]['current_floor'])
                    
                    # Check if episode ended
                    if dones[i]:
                        episode_rewards.append(current_rewards[i])
                        episode_floors.append(max_floors[i])
                        episodes_completed += 1
                        
                        # Reset tracking for this environment
                        current_rewards[i] = 0
                        max_floors[i] = 0
                        
                        # Deactivate environment if we've collected enough episodes
                        if episodes_completed >= num_episodes:
                            active_envs[i] = False
                            active_count -= 1
            
            # Break if all environments are inactive
            if active_count == 0:
                break
                
            # Prepare for next step
            obs = next_obs
            obs_tensor = preprocess_obs_batch(obs, device)
            
    # Close environments
    eval_envs.close()
    
    # Calculate metrics
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    median_reward = np.median(episode_rewards) if episode_rewards else 0
    max_floor = np.max(episode_floors) if episode_floors else 0
    mean_floor = np.mean(episode_floors) if episode_floors else 0
    
    # Set model back to training mode
    model.train()
    
    logger.info(f"Evaluation results: Mean reward: {mean_reward:.2f}, Median reward: {median_reward:.2f}, "
                f"Mean floor: {mean_floor:.2f}, Max floor: {max_floor}")
    
    return mean_reward, max_floor, mean_floor, median_reward

def train_multi_agent(args):
    """
    Train an agent using multiple parallel environments.
    
    Args:
        args: Command-line arguments
    """
    # Create log directory
    log_dir = create_log_dir(args.log_dir, args.run_name)
    logger.info(f"Logging to {log_dir}")
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Initialize TensorBoard
    writer = initialize_tensorboard(log_dir)
    
    # Save command-line arguments
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create vectorized environment
    try:
        logger.info(f"Creating {args.num_envs} environments with worker_id_base={args.worker_id_base}")
        envs = make_vec_envs(
            executable_path=args.env_path, 
            num_envs=args.num_envs, 
            seed=args.seed, 
            timeout=300, 
            realtime_mode=False,
            worker_id_base=args.worker_id_base if hasattr(args, 'worker_id_base') else None
        )
        
        logger.info(f"Created {args.num_envs} environments")
    except Exception as e:
        logger.error(f"Failed to create environments: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Create ActionFlattener for MultiDiscrete action space
    if hasattr(envs.action_space, 'n'):
        action_dim = envs.action_space.n
        action_flattener = None
    else:
        action_flattener = ActionFlattener(envs.action_space.nvec)
        action_dim = action_flattener.action_space.n
    
    logger.info(f"Action space: {envs.action_space}, Action dimension: {action_dim}")
    
    # Create neural network
    input_shape = (12, 84, 84)  # 4 stacked RGB frames (4 * 3 channels)
    
    if args.use_lstm:
        model = RecurrentPPONetwork(
            input_shape=input_shape,
            num_actions=action_dim,
            lstm_hidden_size=args.lstm_hidden_size,
            use_lstm=True
        ).to(device)
        logger.info(f"Created recurrent model with LSTM hidden size {args.lstm_hidden_size}")
    else:
        model = PPONetwork(input_shape=input_shape, num_actions=action_dim).to(device)
        logger.info("Created feedforward model")
    
    # Create PPO agent
    ppo_agent = PPO(
        model=model,
        lr=args.lr,
        clip_eps=args.clip_eps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        epochs=args.epochs,
        batch_size=args.batch_size,
        vf_coef=args.vf_coef,
        ent_reg=args.entropy_reg,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        lr_scheduler='linear',
        adaptive_entropy=True,
        min_entropy=0.01,
        entropy_decay_factor=0.9999,
        update_adv_batch_norm=True,
        entropy_boost_threshold=0.001,
        lr_reset_interval=50,
        use_icm=args.use_icm,
        icm_lr=1e-4,
        icm_reward_scale=0.01,
        icm_forward_weight=0.2,
        icm_inverse_weight=0.8,
        icm_feature_dim=256,
        device=device,
        use_recurrent=args.use_lstm,
        recurrent_seq_len=args.sequence_length
    )
    
    # Initialize ICM if enabled
    if args.use_icm:
        logger.info("Initializing Intrinsic Curiosity Module")
        icm_input_shape = (12, 84, 84)
        ppo_agent.initialize_icm(icm_input_shape, action_dim)
    
    # Load checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            model, metrics, update_count = load_checkpoint(model, args.checkpoint, ppo_agent.optimizer, ppo_agent.scheduler)
            logger.info(f"Loaded checkpoint from {args.checkpoint}")
            optimization_steps = update_count if update_count is not None else 0
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            optimization_steps = 0
    else:
        optimization_steps = 0
    
    # Initialize metrics
    metrics_tracker = MetricsTracker(log_dir)
    
    # Initialize reward history
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_floors = deque(maxlen=100)
    
    # Initialize training loop variables
    start_time = time.time()
    num_updates = args.num_steps // (args.num_envs * args.update_interval)
    steps_per_update = args.num_envs * args.update_interval
    total_steps = 0
    
    # Reset environments
    logger.info("Resetting environments")
    obs = envs.reset()
    obs_tensor = preprocess_obs_batch(obs, device)
    
    # Initialize LSTM states if needed
    lstm_states = None
    if args.use_lstm:
        lstm_states = ppo_agent.model.init_lstm_state(batch_size=args.num_envs, device=device)
    
    # Training loop
    logger.info("Starting training")
    for update in range(num_updates):
        update_start_time = time.time()
        
        # Storage for experiences
        storage = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': [],
            'lstm_states': []
        }
        
        if args.use_lstm:
            # Track which environments need their LSTM states reset
            episode_starts = torch.ones(args.num_envs, dtype=torch.bool, device=device)
            # Initialize or reset LSTM states
            if lstm_states is None or any(episode_starts):
                lstm_states = ppo_agent.model.init_lstm_state(batch_size=args.num_envs, device=device)
        
        # Collect experience
        for step in range(args.update_interval):
            # Store current observations
            storage['obs'].append(obs)
            
            # Store LSTM states if using recurrent policy
            if args.use_lstm and lstm_states is not None:
                # Clone the states to avoid modification issues
                h_state_clone = lstm_states[0].detach().clone()
                c_state_clone = lstm_states[1].detach().clone()
                storage['lstm_states'].append((h_state_clone, c_state_clone))
            
            # Select actions
            try:
                with torch.no_grad():
                    if args.use_lstm:
                        policy_logits, values, next_lstm_states = model(obs_tensor, lstm_states, episode_starts)
                        lstm_states = next_lstm_states
                    else:
                        policy_logits, values = model(obs_tensor)
                    
                    # Sample actions from policy
                    dist = torch.distributions.Categorical(logits=policy_logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    
                    # Convert actions to numpy
                    cpu_actions = actions.cpu().numpy()
                    
                    # Convert actions if using action flattener
                    if action_flattener:
                        env_actions = [action_flattener.lookup_action(a) for a in cpu_actions]
                    else:
                        env_actions = cpu_actions
            except Exception as e:
                logger.error(f"Error during action selection: {e}")
                # Use random actions as fallback
                if action_flattener:
                    env_actions = [action_flattener.lookup_action(np.random.randint(action_dim)) 
                                  for _ in range(args.num_envs)]
                else:
                    env_actions = [np.random.randint(action_dim) for _ in range(args.num_envs)]
                
                # Set placeholder values
                values = torch.zeros(args.num_envs, 1, device=device)
                log_probs = torch.zeros(args.num_envs, device=device)
                cpu_actions = np.zeros(args.num_envs, dtype=np.int64)
            
            # Store values and log probs
            storage['values'].append(values.cpu().numpy())
            storage['log_probs'].append(log_probs.cpu().numpy())
            storage['actions'].append(cpu_actions)
            
            # Take actions in environments
            try:
                next_obs, rewards, dones, infos = envs.step(env_actions)
            except Exception as e:
                logger.error(f"Error during environment step: {e}")
                # Use placeholder values
                next_obs = [None for _ in range(args.num_envs)]
                rewards = np.zeros(args.num_envs)
                dones = np.ones(args.num_envs)  # Assume all environments done on error
                infos = [{"error": True} for _ in range(args.num_envs)]

            # Apply reward shaping if enabled
            if args.reward_shaping:
                for i in range(args.num_envs):
                    # Encourage forward movement
                    if action_flattener and env_actions[i][0] == 1:  # Forward movement
                        rewards[i] += 0.005  # Small bonus for moving forward
                    
                    # Floor completion bonus
                    if 'current_floor' in infos[i] and 'previous_floor' in infos[i]:
                        if infos[i]['current_floor'] > infos[i]['previous_floor']:
                            rewards[i] += 5.0  # Substantial bonus for completing floor
            
            # Store rewards and dones
            storage['rewards'].append(rewards)
            storage['dones'].append(dones)
            
            # Process LSTM states for next step
            if args.use_lstm:
                # Update episode_starts for next step - only where episodes have ended
                episode_starts = torch.tensor(dones, dtype=torch.bool, device=device)
                
                # Reset LSTM states for environments that finished episodes
                if any(dones):
                    # Create a mask for environments that had episodes end
                    done_mask = torch.tensor(dones, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(-1)
                    
                    # Get a clean set of initial states
                    init_states = ppo_agent.model.init_lstm_state(batch_size=args.num_envs, device=device)
                    
                    # Set h state to zero where episodes ended
                    lstm_states = (
                        torch.where(done_mask, init_states[0], lstm_states[0]),
                        torch.where(done_mask, init_states[1], lstm_states[1])
                    )
            
            # Track episode metrics
            for i, info in enumerate(infos):
                if dones[i] or 'episode' in info:
                    ep_info = info.get('episode', {})
                    if not ep_info and 'terminal_observation' in info:
                        # Episode ended naturally
                        ep_reward = info.get('episode_reward', 0)
                        ep_length = info.get('episode_length', 0)
                        ep_floor = info.get('max_floor', 0)
                        
                        if ep_reward is None:
                            # If episode metrics weren't tracked in the environment,
                            # we'll rely on our own tracking
                            continue
                    else:
                        # Use episode info directly
                        ep_reward = ep_info.get('r', 0)
                        ep_length = ep_info.get('l', 0)
                        ep_floor = info.get('current_floor', 0)
                    
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    episode_floors.append(ep_floor)
                    
                    # Log to TensorBoard
                    episode_idx = len(episode_rewards)
                    writer.add_scalar('train/episode_reward', ep_reward, episode_idx)
                    writer.add_scalar('train/episode_length', ep_length, episode_idx)
                    writer.add_scalar('train/max_floor', ep_floor, episode_idx)
                    
                    # Log to console
                    logger.info(f"Episode {episode_idx}: reward={ep_reward:.2f}, length={ep_length}, floor={ep_floor}")
            
            # Update total steps
            total_steps += args.num_envs
            
            # Update observations
            obs = next_obs
            obs_tensor = preprocess_obs_batch(obs, device)
        
        # Calculate returns and advantages
        with torch.no_grad():
            if args.use_lstm:
                _, last_values, _ = model(obs_tensor, lstm_states)
            else:
                _, last_values = model(obs_tensor)
            last_values = last_values.cpu().numpy()
        
        # Convert storage to numpy arrays
        obs_batch = np.array(storage['obs'])
        actions_batch = np.array(storage['actions'])
        rewards_batch = np.array(storage['rewards'])
        dones_batch = np.array(storage['dones'])
        values_batch = np.array(storage['values'])
        log_probs_batch = np.array(storage['log_probs'])
        
        # Calculate returns and advantages
        returns_batch, advantages_batch = [], []
        
        for env_idx in range(args.num_envs):
            env_rewards = rewards_batch[:, env_idx]
            env_values = values_batch[:, env_idx]
            env_dones = dones_batch[:, env_idx]
            
            returns, advantages = ppo_agent.compute_gae(
                rewards=env_rewards,
                values=env_values,
                next_value=last_values[env_idx],
                dones=env_dones
            )
            
            returns_batch.append(returns)
            advantages_batch.append(advantages)
        
        # Transpose to get [env, step] -> [step, env] for easier batch creation
        returns_batch = np.array(returns_batch).transpose()
        advantages_batch = np.array(advantages_batch).transpose()
        
        # Create training batches
        batch_size = args.num_envs * args.update_interval
        num_mini_batches = batch_size // args.batch_size
        
        # Flatten all data
        obs_flat = obs_batch.reshape(-1, *obs_batch.shape[2:])
        actions_flat = actions_batch.flatten()
        log_probs_flat = log_probs_batch.flatten()
        returns_flat = returns_batch.flatten()
        advantages_flat = advantages_batch.flatten()
        
        # Update policy
        if args.use_lstm:
            # Special handling for recurrent policy
            # We need to maintain sequences of data
            ppo_update_info = ppo_agent.update(
                obs_flat, actions_flat, log_probs_flat, returns_flat, advantages_flat, 
                lstm_states=storage['lstm_states'] if args.use_lstm else None
            )
        else:
            # Update for non-recurrent policy
            ppo_update_info = ppo_agent.update(
                obs_flat, actions_flat, log_probs_flat, returns_flat, advantages_flat
            )
        
        # Log update
        optimization_steps += 1
        
        # Update ICM if enabled
        if args.use_icm and optimization_steps % 10 == 0:
            # Sample batch for ICM update
            batch_size = min(1024, len(obs_flat))
            indices = np.random.choice(len(obs_flat), batch_size, replace=False)
            
            icm_obs = torch.tensor(obs_flat[indices], dtype=torch.float32, device=device)
            icm_actions = torch.tensor(actions_flat[indices], dtype=torch.long, device=device)
            icm_next_obs = torch.tensor(obs_flat[indices], dtype=torch.float32, device=device)  # Placeholder
            
            icm_update_info = ppo_agent.update_icm(icm_obs, icm_actions, icm_next_obs)
        
        # Log metrics
        update_end_time = time.time()
        fps = int(args.num_envs * args.update_interval / (update_end_time - update_start_time))
        
        # Log update info
        if len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            avg_floor = np.mean(episode_floors)
            max_floor = np.max(episode_floors) if episode_floors else 0
        else:
            avg_reward = 0
            avg_length = 0
            avg_floor = 0
            max_floor = 0
        
        # Update metrics
        metrics_tracker.update('episode_rewards', avg_reward)
        metrics_tracker.update('episode_lengths', avg_length)
        metrics_tracker.update('episode_floors', avg_floor)
        metrics_tracker.update('policy_losses', ppo_update_info.get('policy_loss', 0))
        metrics_tracker.update('value_losses', ppo_update_info.get('value_loss', 0))
        metrics_tracker.update('entropy_values', ppo_update_info.get('entropy', 0))
        metrics_tracker.update('max_floor_reached', max_floor)
        metrics_tracker.update('steps_per_second', fps)
        
        # Log to TensorBoard
        writer.add_scalar('train/fps', fps, optimization_steps)
        writer.add_scalar('train/policy_loss', ppo_update_info.get('policy_loss', 0), optimization_steps)
        writer.add_scalar('train/value_loss', ppo_update_info.get('value_loss', 0), optimization_steps)
        writer.add_scalar('train/entropy', ppo_update_info.get('entropy', 0), optimization_steps)
        writer.add_scalar('train/approx_kl', ppo_update_info.get('approx_kl', 0), optimization_steps)
        writer.add_scalar('train/clip_fraction', ppo_update_info.get('clip_fraction', 0), optimization_steps)
        writer.add_scalar('train/learning_rate', ppo_agent.optimizer.param_groups[0]['lr'], optimization_steps)
        
        # Log to console
        logger.info(
            f"Update {optimization_steps}/{num_updates} | "
            f"Steps: {total_steps} | "
            f"FPS: {fps} | "
            f"Avg reward: {avg_reward:.2f} | "
            f"Avg length: {avg_length:.2f} | "
            f"Avg floor: {avg_floor:.2f} | "
            f"Max floor: {max_floor} | "
            f"Policy loss: {ppo_update_info.get('policy_loss', 0):.4f} | "
            f"Value loss: {ppo_update_info.get('value_loss', 0):.4f} | "
            f"Entropy: {ppo_update_info.get('entropy', 0):.4f}"
        )
        
        # Save checkpoint periodically
        if optimization_steps % args.save_interval == 0:
            checkpoint_path = os.path.join(log_dir, f"checkpoint_{optimization_steps}.pth")
            save_checkpoint(
                model, 
                checkpoint_path, 
                optimizer=ppo_agent.optimizer, 
                scheduler=ppo_agent.scheduler, 
                metrics=metrics_tracker.metrics,
                update_count=optimization_steps
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save metrics
            metrics_tracker.save(plot=True)
        
        # Run evaluation periodically
        if args.eval_interval > 0 and optimization_steps % args.eval_interval == 0:
            mean_reward, max_floor, mean_floor, median_reward = evaluate_multi_agent(
                model=model,
                env_path=args.env_path,
                num_envs=min(4, args.num_envs),  # Use fewer envs for evaluation
                num_episodes=args.eval_episodes,
                seed=args.seed + 100,
                device=device
            )
            
            # Log evaluation metrics
            writer.add_scalar('eval/mean_reward', mean_reward, optimization_steps)
            writer.add_scalar('eval/max_floor', max_floor, optimization_steps)
            writer.add_scalar('eval/mean_floor', mean_floor, optimization_steps)
            writer.add_scalar('eval/median_reward', median_reward, optimization_steps)
            
            # Save evaluation checkpoint
            eval_checkpoint_path = os.path.join(log_dir, f"eval_checkpoint_{optimization_steps}.pth")
            save_checkpoint(
                model, 
                eval_checkpoint_path, 
                optimizer=ppo_agent.optimizer, 
                scheduler=ppo_agent.scheduler, 
                metrics={
                    'eval_mean_reward': mean_reward,
                    'eval_max_floor': max_floor,
                    'eval_mean_floor': mean_floor,
                    'eval_median_reward': median_reward,
                    **metrics_tracker.metrics
                },
                update_count=optimization_steps
            )
            logger.info(f"Saved evaluation checkpoint to {eval_checkpoint_path}")
    
    # Final evaluation
    logger.info("Training complete. Running final evaluation...")
    mean_reward, max_floor, mean_floor, median_reward = evaluate_multi_agent(
        model=model,
        env_path=args.env_path,
        num_envs=min(4, args.num_envs),
        num_episodes=args.eval_episodes * 2,  # More episodes for final evaluation
        seed=args.seed + 200,
        device=device
    )
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(log_dir, "final_checkpoint.pth")
    save_checkpoint(
        model, 
        final_checkpoint_path, 
        optimizer=ppo_agent.optimizer, 
        scheduler=ppo_agent.scheduler, 
        metrics={
            'final_eval_mean_reward': mean_reward,
            'final_eval_max_floor': max_floor,
            'final_eval_mean_floor': mean_floor,
            'final_eval_median_reward': median_reward,
            **metrics_tracker.metrics
        },
        update_count=optimization_steps
    )
    logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
    
    # Save final metrics
    metrics_tracker.save(plot=True)
    
    # Close environments
    envs.close()
    
    # Close TensorBoard writer
    writer.close()
    
    # Final log
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time / 60:.2f} minutes")
    logger.info(f"Final evaluation - Mean reward: {mean_reward:.2f}, Max floor: {max_floor}")
    
    return model, metrics_tracker.metrics

def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train an agent on Obstacle Tower using multiple parallel environments.")
    
    # Environment
    parser.add_argument("--env_path", type=str, required=True, help="Path to the Obstacle Tower executable")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--reward_shaping", action="store_true", help="Apply reward shaping")
    parser.add_argument("--worker_id_base", type=int, default=None, help="Base worker ID for environments (random if not specified)")
    
    # Training
    parser.add_argument("--num_steps", type=int, default=10_000_000, help="Total number of training steps")
    parser.add_argument("--update_interval", type=int, default=128, help="Steps between PPO updates")
    parser.add_argument("--log_interval", type=int, default=10, help="Updates between logging")
    parser.add_argument("--eval_interval", type=int, default=100, help="Updates between evaluations")
    parser.add_argument("--save_interval", type=int, default=100, help="Updates between saving checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=4, help="Number of PPO epochs per update")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size for PPO updates")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    
    # Model
    parser.add_argument("--use_lstm", action="store_true", help="Use LSTM policy")
    parser.add_argument("--lstm_hidden_size", type=int, default=256, help="LSTM hidden size")
    parser.add_argument("--sequence_length", type=int, default=8, help="Sequence length for LSTM training")
    
    # Optimizer
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--entropy_reg", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm for clipping")
    parser.add_argument("--target_kl", type=float, default=0.03, help="Target KL divergence for early stopping")
    parser.add_argument("--use_icm", action="store_true", help="Use Intrinsic Curiosity Module for exploration")
    
    # System
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--log_dir", type=str, default="logs", help="Base directory for logs")
    parser.add_argument("--run_name", type=str, default=None, help="Name for this run")
    
    return parser.parse_args()

def main():
    """Parse arguments and run training."""
    args = parse_args()
    
    try:
        train_multi_agent(args)
    except Exception as e:
        logger.error(f"Error in training: {e}")
        logger.error(traceback.format_exc())
        raise
        
if __name__ == "__main__":
    main() 