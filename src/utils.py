import numpy as np
import torch

# Generate Random Sample Data
def generate_random_sample_data(num_samples: int):
    """This function generates random sample data for the two-tower model.
    The idea is to understand how to structure the data for the model.

    Args:
        num_samples (int): number of samples

    Returns:
        _type_: _description_
    """
    user_input_dim = 6
    product_input_dim = 3
    user_data = np.random.rand(num_samples, user_input_dim)
    product_data = np.random.rand(num_samples, product_input_dim)
    target_data = np.random.randint(0, 2, size=(num_samples, 1))  # Binary targets for F1 score

    user_tensor = torch.tensor(user_data, dtype=torch.float32)
    product_tensor = torch.tensor(product_data, dtype=torch.float32)
    target_tensor = torch.tensor(target_data, dtype=torch.float32)

    return user_tensor, product_tensor, target_tensor