import torch
from torch.utils.data import Dataset
import random
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base.base_data_loader import BaseDataLoader

# Define regular functions instead of lambdas
def linear_function(x):
    return 1.5 * x + 0.3 + random.uniform(-1, 1)

def quadratic_function(x):
    return 2 * x ** 2 + 0.5 * x + 0.3 + random.uniform(-1, 1)

def harmonic_function(x):
    return 0.5 * x ** 2 + 5 * math.sin(x) + 3 * math.cos(3 * x) + 2 + random.uniform(-1, 1)

class FunctionDataset(Dataset):
    def __init__(self, n_samples=100, function="linear"):
        super().__init__()

        self.n_samples = n_samples
        
        function_map = {
            "linear": linear_function,
            "quadratic": quadratic_function,
            "harmonic": harmonic_function
        }

        if function not in function_map:
            raise ValueError("Function not supported")

        self.function = function_map[function]
        self.x = [random.uniform(0, 2 * math.pi) for _ in range(n_samples)]
        self.y = [self.function(xi) for xi in self.x]
        y_tensor = torch.tensor(self.y, dtype=torch.float32)
        self.mean = torch.mean(y_tensor).item()
        self.std = torch.std(y_tensor).item()
        self.y = ((y_tensor - self.mean) / self.std).tolist()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32).view(1)
        y = torch.tensor(self.y[idx], dtype=torch.float32).view(1)
        return x, y

class FunctionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, function='linear', n_samples=100):
        # Initialize dataset
        dataset = FunctionDataset(n_samples, function)
        # Pass the dataset to the base loader
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)