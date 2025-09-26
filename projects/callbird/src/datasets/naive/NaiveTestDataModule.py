from callbird.src.datasets.naive.NaiveDataModule import NaiveDataModule
from callbird.src.datasets.load_test_dataset import load_test_dataset
from callbird.src.ensure_torch_safe_globals import ensure_torch_safe_globals
from datasets import DatasetDict

# I don't know why this is needed, since it works for others without, but due to the limited time, this is here to stay
ensure_torch_safe_globals()

class NaiveTestDataModule(NaiveDataModule):
    """
    A class to handle the test dataset for Oekofor.
    This class is designed to process the dataset, ensuring it meets the requirements
    of the base class, including renaming columns, constructing file paths, and adding
    necessary columns for event detection.
    """
    
    @property
    def num_classes(self):
        return 106

    def _load_data(self, decode: bool = True) -> DatasetDict:
        dataset = load_test_dataset(self.dataset_config.data_dir)

        # blacklist_naive = readCommentedList("/workspace/projects/callbird/datastats/test/blacklist_naive.txt")
        # dataset = dataset.filter(lambda x: x["ebird_code_and_call"] not in blacklist_naive)

        dataset = super()._process_loaded_multitask_data(dataset, decode)

        return dataset