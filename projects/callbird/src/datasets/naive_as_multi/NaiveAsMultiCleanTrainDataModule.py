from callbird.src.datasets.naive_as_multi.NaiveAsMultiDataModule import NaiveAsMultiDataModule
from callbird.src.datasets.load_train_dataset import load_train_dataset
from callbird.src.ensure_torch_safe_globals import ensure_torch_safe_globals
from callbird.src.readUtils import readCommentedList, readLabeledMapping
from birdset.datamodule import BirdSetDataModule
from datasets import concatenate_datasets, load_dataset, DatasetDict, IterableDataset, IterableDatasetDict, Audio, Features, Value, Dataset
from os import path

# I don't know why this is needed, since it works for others without, but due to the limited time, this is here to stay
ensure_torch_safe_globals()

class NaiveAsMultiCleanTrainDataModule(NaiveAsMultiDataModule):
    """
    A class to handle the train dataset for Oekofor.
    This class is designed to process the dataset, ensuring it meets the requirements
    of the base class, including renaming columns, constructing file paths, and adding
    necessary columns for event detection.
    """
    
    @property
    def num_classes(self):
        return 106

    def _load_data(self, decode: bool = True) -> DatasetDict:
        dataset = load_train_dataset(self.dataset_config.data_dir)

        blacklist_naive = readCommentedList("/workspace/projects/callbird/datastats/train/blacklist_naive.txt")
        dataset = dataset.filter(lambda x: x["ebird_code_and_call"] not in blacklist_naive)

        # Set values in "ebird_code" to "ebird_code_and_call"
        # dataset = dataset.map(lambda x: {"ebird_code_and_call": x["ebird_code"]}, batched=True)

        dataset = super()._process_loaded_multitask_data(dataset, decode)

        return dataset