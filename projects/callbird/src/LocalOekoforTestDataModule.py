from callbird.src.readUtils import readCommentedList, readLabeledMapping
from birdset.datamodule import BirdSetDataModule
from datasets import load_dataset, DatasetDict, IterableDataset, IterableDatasetDict, Audio, Features, Value, Dataset

class LocalOekoforTestDataModule(BirdSetDataModule):
    """
    A class to handle the test dataset for Oekofor.
    This class is designed to process the dataset, ensuring it meets the requirements
    of the base class, including renaming columns, constructing file paths, and adding
    necessary columns for event detection.
    """

    @property
    def num_classes(self):
        return 107
    
    def _load_data(self, decode: bool = True) -> DatasetDict:
        # A list of classes not present in the train set.
        # blacklist_ebird = readCommentedList("/workspace/projects/callbird/datastats/test/blacklist_ebird.txt")
        blacklist_naive = readCommentedList("/workspace/projects/callbird/datastats/test/blacklist_naive.txt")

        dataset = load_dataset(
            "csv",
            data_files = "/workspace/oekofor/testset/labels/*.csv",
            features = Features({ # TODO: Add all features available in BirdSet
                "tsn_code": Value("string"),
                "ebird_code": Value("string"),
                "vocalization_type": Value("string"),
                "start_time": Value("float"),
                "end_time": Value("float"),
                "audio_filename": Value("string"),
            }),
            cache_dir = self.dataset_config.data_dir,
            num_proc = 1,
            trust_remote_code = True, # While not needed for local datasets, it is kept for consistency
        )

        # We need to remove None values from the 'ebird_code' column since the pipeline cannot handle them
        dataset = dataset.map(lambda x: {"ebird_code": x["ebird_code"] if x["ebird_code"] is not None else "NA"}) # TODO: Check if NA is an existing code
        dataset = dataset.map(lambda x: {"vocalization_type": x["vocalization_type"] if x["vocalization_type"] is not None else "NA"}) # TODO: Check if NA is an existing code

        # Load the call type mappings
        calltype_mapping = readLabeledMapping("/workspace/projects/callbird/datastats/call_types_list", "test")
        dataset = dataset.map(lambda x: {"short_call_type": calltype_mapping.get(x["vocalization_type"], None)}) # Using None to force an error if the vocalization type is not found

        # Create naive classes
        dataset = dataset.map(lambda x: {"ebird_code_and_call": f"{x['ebird_code']}_{x['short_call_type']}"})

        # Filter out entries with eBird codes in the blacklist
        dataset = dataset.filter(lambda x: x["ebird_code_and_call"] not in blacklist_naive)



        # Rename 'audio_filename' to 'filepath' to match what the base class expects
        dataset = dataset.rename_column("audio_filename", "filepath")

        # Setting absolute paths for the audio files
        def update_filepath(example):
            example["filepath"] = f"/workspace/oekofor/testset/audio_files/{example['filepath']}.flac"
            return example

        dataset = dataset.map(update_filepath)

        def add_event_columns(example):
            example["detected_events"] = (example["start_time"], example["end_time"])
            # TODO: Fix event_cluster value
            example["event_cluster"] = [0]
            return example

        dataset = dataset.map(add_event_columns)

        if isinstance(dataset, IterableDataset | IterableDatasetDict):
            print("Iterable datasets not supported yet.")
            # log.error("Iterable datasets not supported yet.")
            return

        assert isinstance(dataset, DatasetDict | Dataset)
        dataset = self._ensure_train_test_splits(dataset)
        def add_multilabel_column(example):
            example["ebird_code"] = example["ebird_code_and_call"] # TODO: Check if this is correct / needed
            example["ebird_code_multilabel"] = example["ebird_code_and_call"] # TODO: Check if this is correct / needed
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
        return dataset