from callbird.src.readUtils import readCommentedList, readLabeledMapping
from birdset.datamodule import BirdSetDataModule
from datasets import concatenate_datasets, load_dataset, DatasetDict, IterableDataset, IterableDatasetDict, Audio, Features, Value, Dataset
from os import path
from birdset import utils

log = utils.get_pylogger(__name__)

class NaiveDataModule(BirdSetDataModule):
    """
    A class to handle the train dataset for Oekofor.
    This class is designed to process the dataset, ensuring it meets the requirements
    of the base class, including renaming columns, constructing file paths, and adding
    necessary columns for event detection.
    """

    @property
    def num_classes(self):
        pass
    
    def _process_loaded_naive_data(self, dataset, decode: bool = True):
        def add_event_columns(example):
            example["detected_events"] = (example["start_time"], example["end_time"])
            # TODO: Fix event_cluster value
            example["event_cluster"] = [0]
            return example

        dataset = dataset.map(add_event_columns)

        if isinstance(dataset, IterableDataset | IterableDatasetDict):
            log.error("Iterable datasets not supported yet.")
            return

        assert isinstance(dataset, DatasetDict | Dataset)
        dataset["test"] = dataset["train"]
        # dataset = self._ensure_train_test_splits(dataset)
        def add_multilabel_column(example):
            example["ebird_code"] = example["ebird_code"] # TODO: Check if this is correct / needed
            example["ebird_code_multilabel"] = example["ebird_code"] # TODO: Check if this is correct / needed
            return example
        
        dataset = dataset.map(add_multilabel_column)

        labels = set()
        for split in dataset.keys():
            labels.update(dataset[split]["ebird_code_multilabel"])
        labels = sorted(labels)  # Sort to ensure consistent ordering
        label_to_id = {lbl: i for i, lbl in enumerate(labels)}

        def label_to_id_fn(batch):
            for i in range(len(batch["ebird_code_multilabel"])):
                batch["ebird_code_multilabel"][i] = label_to_id[batch["ebird_code_multilabel"][i]]
            return batch

        dataset = dataset.map(
            label_to_id_fn,
            batched=True,
            batch_size=500,
            load_from_cache_file=True,
            num_proc=self.dataset_config.n_workers,
        )

        if "test" in dataset:
           dataset["test_5s"] = dataset["test"]

        if self.dataset_config.subset:
            dataset = self._fast_dev_subset(dataset, self.dataset_config.subset)

        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.dataset_config.sample_rate,
                mono=True,
                decode=decode,
            ),
        )

        print(f"Classes: {dataset.unique('ebird_code_multilabel')}")
        print(f"Classes raw: {dataset.unique('ebird_code')}")

        return dataset