import numpy as np


def train_val_test_split(chunk, val_size = 0.15, test_size=0.15, random_state=42):
    '''
    function to split dataset into train, validation and test sets.
    This function is called using multiple chunks in parallel
    '''
    idxs = chunk.index.to_numpy().copy()
    test_count = int(len(chunk) * test_size)
    val_count = int(len(chunk) * val_size)
    np.random.seed(random_state)
    test_idx = np.random.choice(idxs, test_count, replace=False)
    train_idx = np.setdiff1d(idxs, test_idx)
    np.random.seed(random_state)
    val_idx = np.random.choice(train_idx, val_count, replace=False)
    train_idx = np.setdiff1d(train_idx, val_idx)
    return train_idx, val_idx, test_idx