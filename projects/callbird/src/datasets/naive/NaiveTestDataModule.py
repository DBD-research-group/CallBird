from callbird.src.datasets.load_test_dataset import load_test_dataset
from callbird.src.ensure_torch_safe_globals import ensure_torch_safe_globals
from projects.callbird.src.datasets.naive.NaiveDataModule import NaiveDataModule
from datasets import DatasetDict

# I don't know why this is needed, since it works for others without, but due to the limited time, this is here to stay
ensure_torch_safe_globals()

class NaiveTestDataModule(NaiveDataModule):

    @property
    def num_classes(self):
        return 54
    
    def _load_data(self, decode: bool = True) -> DatasetDict:
        dataset = load_test_dataset(self.dataset_config.data_dir)

        dataset["test"] = dataset["train"]
        dataset = super()._process_loaded_naive_data(dataset, decode)

        dataset["test"] = dataset["train"]

        return dataset