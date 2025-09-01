from birdset import utils
from birdset.datamodule import BirdSetDataModule
from datasets import DatasetDict, IterableDataset, IterableDatasetDict, Audio, Dataset

log = utils.get_pylogger(__name__)

class NaiveAsMultiDataModule(BirdSetDataModule):

    @property
    def num_classes(self):
        return 198
    
    def _process_loaded_multitask_data(self, dataset, decode: bool = True):
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
        dataset = self._ensure_train_test_splits(dataset)
        def add_multilabel_column(example):
            example["ebird_code_multilabel"] = example["ebird_code_and_call"]
            return example
        
        dataset = dataset.map(add_multilabel_column)

        ebird_labels = set()
        for split in dataset.keys():
            ebird_labels.update(dataset[split]["ebird_code_and_call"])

        self.ebird_labels = sorted(list(ebird_labels))

        ebird_label_to_id = {lbl: i for i, lbl in enumerate(self.ebird_labels)}

        def label_to_id_fn(batch):
            for i in range(len(batch["ebird_code_multilabel"])):
                batch["ebird_code_multilabel"][i] = ebird_label_to_id[batch["ebird_code_multilabel"][i]]

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

        # print length of ebird_labels
        print(f"Number of unique eBird labels: {self.num_classes}")

        return dataset

    def _preprocess_data(self, dataset):
        if self.dataset_config.task == "multilabel":
             # pick only train and test_5s dataset
            dataset = DatasetDict(
                {split: dataset[split] for split in ["train", "test_5s"]}
            )

            log.info(">> Mapping train data.")
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                num_proc=self.dataset_config.n_workers,
                desc="Train event mapping",
            )

            log.info(">> One-hot-encode classes")
            for split in ["train", "test_5s"]:
                dataset[split] = dataset[split].map(
                    self._classes_one_hot,
                    fn_kwargs={
                        "label_column_name": "ebird_code_multilabel",
                        "num_classes": self.num_classes
                    },
                    batched=True,
                    batch_size=300,
                    num_proc=self.dataset_config.n_workers,
                    desc=f"One-hot-encoding ebird labels for {split}.",
                )

            dataset = dataset.rename_column("ebird_code_multilabel", "labels_ebird")

            dataset_test = dataset.pop("test_5s")
            dataset["test"] = dataset_test
        else:
            raise f"{self.dataset_config.task=} is not supported, choose (multilabel, multiclass)"

        for split in ["train", "test"]:
            dataset[split] = dataset[split].select_columns(
                ["filepath", "labels_ebird", "detected_events", "start_time", "end_time"]
            )

        return dataset