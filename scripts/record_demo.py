from obstacle_tower_env import ObstacleTowerEnv
import numpy as np

def record_demos():
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower')
    obs = env.reset()
    demos = []

    while True:
        env.render()
        action = int(input("Enter action (0-5): "))  # Adjust based on action space
        next_obs, reward, done, _ = env.step(action)
        demos.append((obs, action))
        obs = next_obs
        if done:
            break
    
    np.save("data/demos.npy", demos)
    env.close()

if __name__ == "__main__":
    record_demos()