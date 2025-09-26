from callbird.src.datasets.multi.MultiDataModule import MultiDataModule
from callbird.src.datasets.load_train_dataset import load_train_dataset
from callbird.src.ensure_torch_safe_globals import ensure_torch_safe_globals
from callbird.src.readUtils import readCommentedList, readLabeledMapping
from birdset.datamodule import BirdSetDataModule
from datasets import concatenate_datasets, load_dataset, DatasetDict, IterableDataset, IterableDatasetDict, Audio, Features, Value, Dataset
from os import path

# I don't know why this is needed, since it works for others without, but due to the limited time, this is here to stay
ensure_torch_safe_globals()

class MultiTrainDataModule(MultiDataModule):
    """
    A class to handle the train dataset for Oekofor.
    This class is designed to process the dataset, ensuring it meets the requirements
    of the base class, including renaming columns, constructing file paths, and adding
    necessary columns for event detection.
    """
    
    def _load_data(self, decode: bool = True) -> DatasetDict:
        dataset = load_train_dataset(self.dataset_config.data_dir)

        # list all birds that have "call_type" "ac_b (Alarmruf_Bodenfeinde)"
        # dataset = dataset.filter(lambda x: x["call_type"] == "ac_b (Alarmruf_Bodenfeinde)")
        # ebird_code_list = dataset["train"].unique("ebird_code")
        # print(f"Unique ebird codes in the filtered dataset: {ebird_code_list}")
        # raise NotImplementedError("This code snippet is for illustration only. Please implement the logic as needed.")

        # dataset = dataset.filter(lambda x: x["ebird_code"] == "eurbla")
        # dataset = dataset.filter(lambda x: x["call_type"] == "ac_b (Alarmruf_Bodenfeinde)")

        dataset = super()._process_loaded_multitask_data(dataset, decode)

        return dataset