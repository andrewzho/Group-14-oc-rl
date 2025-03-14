# Multi-Agent Reinforcement Learning for Obstacle Tower

This folder contains an implementation of multi-agent reinforcement learning for the Obstacle Tower environment. It allows training with multiple parallel environments, which significantly speeds up the training process.

## Components

- `env_wrapper.py`: Contains the wrapper for the Obstacle Tower environment that makes it compatible with the parallel environment runner.
- `vec_env.py`: Contains the implementation of a vectorized environment that runs multiple environments in parallel processes.
- `train_multi.py`: The main training script that uses the vectorized environment for faster training.

## Benefits of Multi-Agent Training

- **Faster Data Collection**: Collecting experiences from multiple environments in parallel significantly speeds up training.
- **More Diverse Experiences**: Multiple environments provide more diverse experiences for the agent to learn from.
- **Improved Exploration**: Multiple agents can explore different parts of the environment simultaneously.
- **Better Generalization**: Training on multiple instances of the environment can lead to better generalization.

## Usage

To train with multiple environments, use the `train_multi.py` script:

```bash
python -m multi_agent.train_multi \
    --env_path=path/to/ObstacleTower.exe \
    --num_envs=8 \
    --num_steps=10000000 \
    --use_lstm \
    --lstm_hidden_size=256 \
    --sequence_length=8 \
    --reward_shaping \
    --save_interval=20
```

### Key Parameters

- `--env_path`: Path to the Obstacle Tower executable
- `--num_envs`: Number of parallel environments to use (e.g., 4, 8, 16)
- `--use_lstm`: Use LSTM-based recurrent policy
- `--reward_shaping`: Enable reward shaping for better learning
- `--use_icm`: Enable Intrinsic Curiosity Module for exploration

## Requirements

This implementation requires the same dependencies as the main project, plus:

- Python 3.7+
- PyTorch 1.7+
- NumPy
- TensorBoard

## Performance Considerations

- The number of environments you can run in parallel depends on your hardware. Start with a smaller number (e.g., 4) and increase if your system can handle it.
- Using too many environments may cause memory issues, especially with GPUs.
- For optimal performance, the number of environments should generally be a multiple of the number of CPU cores available.

## Logging and Monitoring

The script logs training progress to TensorBoard, which can be viewed by running:

```bash
tensorboard --logdir=logs/multi_agent
```

Training metrics and checkpoints are saved to the specified log directory. The checkpoints can be loaded for continued training or evaluation. 