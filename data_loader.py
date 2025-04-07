import torch
from torchvision import datasets, transforms
import numpy as np

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def partition_mnist_noniid(train_dataset, num_clients=10, shards_per_client=2):
    # num_shards = num_clients * shards_per_client
    num_shards = 100
    shard_size = len(train_dataset) // num_shards
    data_indices = np.arange(len(train_dataset))
    labels = np.array(train_dataset.targets)

    sorted_indices = data_indices[np.argsort(labels)]

    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
    np.random.shuffle(shards)

    client_data_map = {i: [] for i in range(num_clients)}
    for i in range(num_clients):
        assigned_shards = shards[i * shards_per_client:(i + 1) * shards_per_client]
        client_data_map[i] = np.concatenate(assigned_shards)

    return client_data_map
