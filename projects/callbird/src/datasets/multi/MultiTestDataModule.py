from os import path
from callbird.src.data.multi.MultiDataModule import MultiDataModule
from callbird.src.data import load_test_dataset
from callbird.src.ensure_torch_safe_globals import ensure_torch_safe_globals
from callbird.src.readUtils import readCommentedList, readLabeledMapping
from birdset.datamodule import BirdSetDataModule
from datasets import load_dataset, DatasetDict, Features, Value

# I don't know why this is needed, since it works for others without, but due to the limited time, this is here to stay
ensure_torch_safe_globals()

class MultiTestDataModule(MultiDataModule):
    """
    A class to handle the test dataset for Oekofor.
    This class is designed to process the dataset, ensuring it meets the requirements
    of the base class, including renaming columns, constructing file paths, and adding
    necessary columns for event detection.
    """
    
    def _load_data(self, decode: bool = True) -> DatasetDict:
        dataset = load_test_dataset(self.dataset_config.data_dir)

        dataset = super()._process_loaded_multitask_data(dataset, decode)

        return dataset