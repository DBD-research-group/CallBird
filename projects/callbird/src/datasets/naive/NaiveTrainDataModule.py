from callbird.src.data import load_train_dataset
from projects.callbird.src.data.naive.NaiveDataModule import NaiveDataModule
from datasets import DatasetDict

class NaiveTrainDataModule(NaiveDataModule):

    @property
    def num_classes(self):
        return 198
    
    def _load_data(self, decode: bool = True) -> DatasetDict:
        dataset = load_train_dataset(self.dataset_config.data_dir)

        dataset = super()._process_loaded_naive_data(dataset, decode)

        return dataset