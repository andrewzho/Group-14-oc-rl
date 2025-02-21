import numpy as np
import torch

def normalize(x):
    x = np.array(x)
    return (x - x.mean()) / (x.std() + 1e-8)

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")