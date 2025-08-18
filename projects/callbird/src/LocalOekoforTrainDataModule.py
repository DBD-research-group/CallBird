from callbird.src.readUtils import readCommentedList, readLabeledMapping
from birdset.datamodule import BirdSetDataModule
from datasets import load_dataset, DatasetDict, IterableDataset, IterableDatasetDict, Audio, Features, Value, Dataset
from os import path

class LocalOekoforTrainDataModule(BirdSetDataModule):
    """
    A class to handle the train dataset for Oekofor.
    This class is designed to process the dataset, ensuring it meets the requirements
    of the base class, including renaming columns, constructing file paths, and adding
    necessary columns for event detection.
    """

    @property
    def num_classes(self):
        return 200
    
    def _load_data(self, decode: bool = True) -> DatasetDict:
        blacklist_naive = readCommentedList("/workspace/projects/callbird/datastats/train/blacklist_naive.txt")

        dataset = load_dataset(
            "csv",
            data_files = "/workspace/oekofor/trainset/csvlabels/*.csv",
            features = Features({ # TODO: Add all features available in BirdSet
                "ebird_code": Value("string"),
                "call_type": Value("string"),
                "start_sample [s]": Value("float"),
                "end_sample [s]": Value("float"),
                "actual_filename": Value("string"),
            }),
            delimiter=";",
            cache_dir = self.dataset_config.data_dir,
            num_proc = 1,
            trust_remote_code = True, # While not needed for local datasets, it is kept for consistency
        )

        # We need to remove None values from the 'ebird_code' column since the pipeline cannot handle them
        dataset = dataset.map(lambda x: {"ebird_code": x["ebird_code"] if x["ebird_code"] is not None else "NA"})
        dataset = dataset.map(lambda x: {"call_type": x["call_type"] if x["call_type"] is not None else "NA"})

        # Load the call type mappings
        calltype_mapping = readLabeledMapping("/workspace/projects/callbird/datastats/call_types_list", "train")
        dataset = dataset.map(lambda x: {"short_call_type": calltype_mapping.get(x["call_type"], None)}) # Using None to force an error if the call type is not found

        # Create naive classes
        dataset = dataset.map(lambda x: { "ebird_code_and_call": f"{x['ebird_code']}_{x['short_call_type']}" })

        # Filter out entries with eBird codes in the blacklist
        dataset = dataset.filter(lambda x: x["ebird_code_and_call"] not in blacklist_naive)


        dataset = dataset.rename_column("start_sample [s]", "start_time")
        dataset = dataset.rename_column("end_sample [s]", "end_time")
        dataset = dataset.rename_column("actual_filename", "filepath")

        files_blacklist = readCommentedList("/workspace/projects/callbird/datastats/train/blacklist_files.txt")
        # Filter out entries with file paths in the blacklist
        dataset = dataset.filter(lambda x: x["filepath"] not in files_blacklist)

        # Setting absolute paths for the audio files
        def update_filepath(example):
            example["filepath"] = f"/workspace/oekofor/dataset/{example['filepath']}.flac"

            # if not path.exists(example["filepath"]):
            #     example["filepath"] = example["filepath"].replace(".flac", ".wav")

            return example

        dataset = dataset.map(update_filepath)

        dataset = dataset.filter(lambda x: path.exists(x["filepath"]))

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

        print(f"Classes: {dataset.unique('ebird_code_multilabel')}")
        print(f"Classes raw: {dataset.unique('ebird_code')}")

        return dataset