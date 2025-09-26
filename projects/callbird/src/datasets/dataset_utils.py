from datasets import IterableDatasetDict, concatenate_datasets

def reduce_samples(dataset: IterableDatasetDict, column_name: str, target_value: str, max_samples: int):
    """
    Performaned downsampling method to reduce the number of samples for a specific column value in a dataset.

    Args:
        dataset (DatasetDict): The dataset to be downsampled.
        column_name (str): The name of the column to be downsampled.
        target_value (str): The specific value in the column to be downsampled.
        max_samples (int): The maximum number of samples to retain for the target value.
    Returns:
        DatasetDict: The downsampled dataset.
    """
    
    # Create own dataset only containing target column samples
    target_column_dataset = dataset.filter(lambda x: x[column_name] == target_value)
    # Create dataset containing all other samples
    other_dataset = dataset.filter(lambda x: x[column_name] != target_value)
    # Downsample the target column dataset to the specified number of samples
    target_column_dataset = target_column_dataset['train'].shuffle(seed=42).select(range(min(max_samples, len(target_column_dataset['train']))))
    # Merge the downsampled target column dataset with the rest of the dataset
    dataset['train'] = concatenate_datasets([other_dataset['train'], target_column_dataset])
    return dataset