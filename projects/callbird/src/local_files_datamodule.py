from birdset.datamodule import BirdSetDataModule
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    Audio,
    load_dataset,
    load_dataset_builder,
    Features,
    Value,
)

class LocalFilesDataModule(BirdSetDataModule):
    """
    """
    
    @property
    def num_classes(self):
        # TODO: Implement a dynamic way to determine the number of classes
        return 60 # Result of ```len(dataset["train"].unique("ebird_code"))``` after loading the dataset.

    def _load_data(self, decode: bool = True) -> DatasetDict:
        dataset_args = {
            # "path": self.dataset_config.hf_path,
            "cache_dir": self.dataset_config.data_dir,
            "num_proc": 3,
            "trust_remote_code": True,
        }

        if self.dataset_config.hf_name != "esc50":  # special esc50 case due to naming
            dataset_args["name"] = self.dataset_config.hf_name

        # By defining the feature type for 'tsn_code' as 'string', we prevent
        # the automatic type inference from causing conflicts between files.
        # The other columns are left as None to be inferred automatically.
        custom_features = Features(
            {
                "tsn_code": Value("string"),
                "ebird_code": Value("string"),
                "start_time": Value("float"),
                "end_time": Value("float"),
                "audio_filename": Value("string"),
            }
        )
        # dataset = load_dataset(**dataset_args)
        # print current working directory
        import os
        print(os.getcwd())
        dataset = load_dataset(
            "csv",
            data_files="/workspace/oekofor/testset/labels/*_2020*.csv",
            features=custom_features,
            **dataset_args,
        )

        dataset = dataset.filter(lambda x: x["ebird_code"] is not None)

        # Rename 'audio_filename' to 'filepath' to match what the base class expects.
        dataset = dataset.rename_column("audio_filename", "filepath")

        def update_filepath(example):
            """
            Constructs the full path to the audio file.
            The CSV provides the filename, but not the path or extension.
            This function prepends the directory and appends the extension.
            """
            audio_dir = "/workspace/oekofor/testset/audio_files/"
            example["filepath"] = f"{audio_dir}{example['filepath']}.flac"
            return example

        dataset = dataset.map(update_filepath)

        def add_required_columns(example):
            """
            Adds the 'detected_events' and 'event_cluster' columns.
            'detected_events' is created from start and end times.
            'event_cluster' is set to a default value of [0], as we assume
            each row represents a single, valid event cluster.
            """
            example["detected_events"] = (example["start_time"], example["end_time"])
            example["event_cluster"] = [0] # Add the event_cluster column
            return example

        dataset = dataset.map(add_required_columns)

        if isinstance(dataset, IterableDataset | IterableDatasetDict):
            print("Iterable datasets not supported yet.")
            # log.error("Iterable datasets not supported yet.")
            return
        
        assert isinstance(dataset, DatasetDict | Dataset)
        dataset = self._ensure_train_test_splits(dataset)
        def add_multilabel_column(example):
            example["ebird_code_multilabel"] = example["ebird_code"]
            # example["labels"] = example["ebird_code"]
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