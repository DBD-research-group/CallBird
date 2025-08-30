from callbird.src.data.multi.MultiDataModule import MultiDataModule
from callbird.src.data.load_train_dataset import load_train_dataset
from callbird.src.readUtils import readCommentedList, readLabeledMapping
from birdset.datamodule import BirdSetDataModule
from datasets import concatenate_datasets, load_dataset, DatasetDict, IterableDataset, IterableDatasetDict, Audio, Features, Value, Dataset
from os import path

class MultiTrainDataModule(MultiDataModule):
    """
    A class to handle the train dataset for Oekofor.
    This class is designed to process the dataset, ensuring it meets the requirements
    of the base class, including renaming columns, constructing file paths, and adding
    necessary columns for event detection.
    """
    
    def _load_data(self, decode: bool = True) -> DatasetDict:
        dataset = load_train_dataset(self.dataset_config.data_dir)

        dataset = super()._process_loaded_multitask_data(dataset, decode)

        return dataset