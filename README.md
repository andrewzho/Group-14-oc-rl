# Obstacle Tower RL Project

This project trains a reinforcement learning (RL) agent to navigate the Obstacle Tower environment using the Proximal Policy Optimization (PPO) algorithm. The agent learns to solve procedurally generated levels by interacting with the environment and receiving rewards for progress.

The code supports both a custom PPO implementation and an optional integration with Stable-Baselines3 for easier experimentation.

## Prerequisites

Before setting up the project, ensure you have the following installed on your system:

* Python 3.7 or higher ([Download Python](https://www.python.org/downloads/))
* Conda ([Install Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html))
* Git (for cloning the repository - optional but recommended)
* Unity ML-Agents (required for the Obstacle Tower environment)

Additionally, you will need to download the Obstacle Tower executable from the [Obstacle Tower Releases](https://github.com/Unity-Technologies/obstacle-tower-env/releases) page. Place the executable in the `ObstacleTower/` directory within your project folder.

## Project Structure

```
Group-14-oc-rl/
│
├── ObstacleTower/               # Directory for the Obstacle Tower executable
│   └── obstacletower            # Place the executable here
│
├── src/
│   ├── train.py                 # Main training script (custom PPO)
│   ├── train2.py                # Optional: Training with Stable-Baselines3
│   ├── model.py                 # Neural network architecture
│   ├── ppo.py                   # Custom PPO algorithm implementation
│   ├── utils.py                 # Utility functions (e.g., normalization, saving)
│   └── obstacle_tower_env.py    # Obstacle Tower environment wrapper
│
├── logs/                        # Directory for saving checkpoints and logs
│
├── requirements.txt             # Full list of Python packages
│
└── README.md                    # This file
```

## Environment Setup

### 1. Clone the Repository (Optional)

If the project is hosted on a version control system like Git, clone it to your local machine:

```bash
git clone https://github.com/your-repo/Group-14-oc-rl.git
cd Group-14-oc-rl
```

Alternatively, download the project files manually.

### 2. Create a Conda Environment

This project uses a Conda environment for dependency management. Create and activate it as follows:

```bash
conda create -n new-175-rl python=3.8   
conda activate new-175-rl
```

### 3. Install Dependencies

Install the essential packages required to run the project:

```bash
pip install -e .
pip install protobuf==3.20.3
pip install mlagents-envs==0.17.0
pip install torch
```

### 4. Download and Place the Obstacle Tower Executable

Download the Obstacle Tower executable for your operating system and place it in the `ObstacleTower/` directory:

| Platform       | Download Link |
|---------------|---------------|
| Linux (x86_64) | [obstacletower_v4.1_linux.zip](https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_linux.zip) |
| Mac OS X       | [obstacletower_v4.1_osx.zip](https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_osx.zip) |
| Windows        | [obstacletower_v4.1_windows.zip](https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_windows.zip) |

## Running the Code

### 1. Activate the Conda Environment

```bash
# Windows
conda activate new-175-rl

# macOS/Linux
conda activate new-175-rl
```

### 2. Navigate to the Project Directory

```bash
cd path/to/Group-14-oc-rl
```

### 3. Run the Training Script

To train the agent using the custom PPO implementation:

```bash
python -m src.train --realtime
```

* `--realtime`: Optional flag to enable real-time visualization of the agent's actions in the Unity window

Alternatively, to use the Stable-Baselines3 version:

```bash
python -m src.train2 --realtime
```

### 4. Monitor Training

* The Unity window will open, showing the agent interacting with the environment
* Training logs (e.g., episode rewards) will be printed to the console
* Checkpoints are saved periodically in the `logs/checkpoints/` directory

## Technical Details

### Environment

The Obstacle Tower environment is a procedurally generated 3D tower with increasing difficulty. The agent must navigate floors, avoid obstacles, and solve puzzles to progress.

### Implementation Details

* **Observation Preprocessing**: Observations are resized to (84, 84, 3), transposed to (3, 84, 84), and normalized to [0, 1] for the neural network
* **Action Space**: The environment uses a MultiDiscrete action space with four components (movement, rotation, jump, and interaction), which is flattened into a single discrete space for the policy
* **PPO Algorithm**: The agent uses Proximal Policy Optimization to learn a policy that maximizes cumulative rewards. The custom implementation includes frame stacking, reward shaping, and entropy regularization to improve learning

## Troubleshooting

Common issues and their solutions:

* **Missing Dependencies**: Ensure all required packages are installed. Run `conda list` or `pip list` to check
* **Incorrect Executable Path**: Make sure the Obstacle Tower executable is in the `ObstacleTower/` directory and named correctly (e.g., `obstacletower.exe` on Windows)
* **Unity Window Not Opening**: Check if the executable path is correct and that your system supports graphical output
* **CPU vs. GPU Usage**: If you have a GPU but the code uses CPU, ensure PyTorch is installed with CUDA support (e.g., via `conda install pytorch ... -c pytorch`)
* **Seed Warnings**: The environment may warn about invalid seeds; this is normal and will use a random seed within the valid range

For additional help, refer to the [Obstacle Tower documentation](https://github.com/Unity-Technologies/obstacle-tower-env) or the [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
