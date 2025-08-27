from callbird.src.readUtils import readCommentedList, readLabeledMapping
from birdset.datamodule import BirdSetDataModule
from datasets import concatenate_datasets, load_dataset, DatasetDict, IterableDataset, IterableDatasetDict, Audio, Features, Value, Dataset
from os import path

class LocalOekoforTrainDataModule(BirdSetDataModule):
    """
    A class to handle the train dataset for Oekofor.
    This class is designed to process the dataset, ensuring it meets the requirements
    of the base class, including renaming columns, constructing file paths, and adding
    necessary columns for event detection.
    """

    @property
    def num_classes_ebird(self):
        return len(self.ebird_labels)

    @property
    def num_classes_calltype(self):
        return len(self.calltype_labels)

    @property
    def num_classes(self):
        return 56
    
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

        # Limit the number of "NA" ebird_code entries to 5000
        na_dataset = dataset.filter(lambda x: x["ebird_code"] == "NA")
        other_dataset = dataset.filter(lambda x: x["ebird_code"] != "NA")
        na_subset = na_dataset['train'].shuffle(seed=42).select(range(min(5000, len(na_dataset['train']))))
        dataset['train'] = concatenate_datasets([other_dataset['train'], na_subset])
        
        # Load the call type mappings
        calltype_mapping = readLabeledMapping("/workspace/projects/callbird/datastats/call_types_list", "train")
        dataset = dataset.map(lambda x: {"short_call_type": calltype_mapping.get(x["call_type"], None)}) # Using None to force an error if the call type is not found

        # Create naive classes
        dataset = dataset.map(lambda x: { "ebird_code_and_call": f"{x['ebird_code']}_{x['short_call_type']}" })

        # Filter out entries with eBird codes in the blacklist
        # We dont filter this now, to include all possible classes
        # dataset = dataset.filter(lambda x: x["ebird_code_and_call"] not in blacklist_naive)


        dataset = dataset.rename_column("start_sample [s]", "start_time")
        dataset = dataset.rename_column("end_sample [s]", "end_time")
        dataset = dataset.rename_column("actual_filename", "filepath")

        files_blacklist = readCommentedList("/workspace/projects/callbird/datastats/train/blacklist_files.txt")
        # Filter out entries with file paths in the blacklist
        dataset = dataset.filter(lambda x: x["filepath"] not in files_blacklist)

        # Setting absolute paths for the audio files
        def update_filepath(example):
            example["filepath"] = f"/workspace/oekofor/dataset/{example['filepath']}.flac"

            if not path.exists(example["filepath"]):
                example["filepath"] = example["filepath"].replace(".flac", ".wav")

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
            example["ebird_code_multilabel"] = example["ebird_code"]
            example["call_type_multilabel"] = example["short_call_type"]

            return example
        
        dataset = dataset.map(add_multilabel_column)

        ebird_labels = set()
        calltype_labels = set()

        for split in dataset.keys():
            ebird_labels.update(dataset[split]["ebird_code"])
            calltype_labels.update(dataset[split]["short_call_type"])

        self.ebird_labels = sorted(list(ebird_labels))
        self.calltype_labels = sorted(list(calltype_labels))

        ebird_label_to_id = {lbl: i for i, lbl in enumerate(self.ebird_labels)}
        calltype_label_to_id = {lbl: i for i, lbl in enumerate(self.calltype_labels)}

        def label_to_id_fn(batch):
            for i in range(len(batch["ebird_code_multilabel"])):
                batch["ebird_code_multilabel"][i] = ebird_label_to_id[batch["ebird_code_multilabel"][i]]

            for i in range(len(batch["call_type_multilabel"])):
                batch["call_type_multilabel"][i] = calltype_label_to_id[batch["call_type_multilabel"][i]]

            ## batch["ebird_code_multilabel"] = [ebird_label_to_id[label] for label in batch["ebird_code_multilabel"]]
            ## batch["call_type_multilabel"] = [calltype_label_to_id[label] for label in batch["call_type_multilabel"]]
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

        # print(f"Classes: {dataset.unique('ebird_code_multilabel')}")
        # print(f"Classes raw: {dataset.unique('ebird_code')}")

        return dataset

    def _preprocess_data(self, dataset):
        if self.dataset_config.task == "multilabel":
             # pick only train and test_5s dataset
            dataset = DatasetDict(
                {split: dataset[split] for split in ["train", "test_5s"]}
            )

            print(">> Mapping train data.")
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                num_proc=self.dataset_config.n_workers,
                desc="Train event mapping",
            )

            print(">> One-hot-encode classes")
            for split in ["train", "test_5s"]:
                dataset[split] = dataset[split].map(
                    self._classes_one_hot,
                    fn_kwargs={
                        "label_column_name": "ebird_code_multilabel",
                        "num_classes": self.num_classes_ebird
                    },
                    batched=True,
                    batch_size=300,
                    num_proc=self.dataset_config.n_workers,
                    desc=f"One-hot-encoding ebird labels for {split}.",
                )
                dataset[split] = dataset[split].map(
                    self._classes_one_hot,
                    fn_kwargs={
                        "label_column_name": "call_type_multilabel",
                        "num_classes": self.num_classes_calltype
                    },
                    batched=True,
                    batch_size=300,
                    num_proc=self.dataset_config.n_workers,
                    desc=f"One-hot-encoding calltype labels for {split}.",
                )

            dataset = dataset.rename_column("ebird_code_multilabel", "labels_ebird")
            dataset = dataset.rename_column("call_type_multilabel", "labels_calltype")

            dataset_test = dataset.pop("test_5s")
            dataset["test"] = dataset_test
        else:
            raise f"{self.dataset_config.task=} is not supported, choose (multilabel, multiclass)"

        for split in ["train", "test"]:
            dataset[split] = dataset[split].select_columns(
                ["filepath", "labels_ebird", "labels_calltype", "detected_events", "start_time", "end_time"]
            )

        return dataset