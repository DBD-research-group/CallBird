from birdset import utils
from birdset.datamodule import BirdSetDataModule
from datasets import DatasetDict, IterableDataset, IterableDatasetDict, Audio, Dataset

from datasets import DatasetDict

from birdset import utils
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.components.event_mapping import XCEventMapping
from birdset.configs import DatasetConfig, LoadersConfig

log = utils.get_pylogger(__name__)

class MultiDataModule(BirdSetDataModule):

    def __init__(
        self,
        calltype_map: str,
        filter_naive: bool,
        unknown_ebird_code: str,
        num_combined_classes: int,
        limit_samples: bool,
        filter_unspecified: bool,
        dataset: DatasetConfig = DatasetConfig(
            data_dir="data_birdset/HSN",
            hf_path="DBD-research-group/BirdSet",
            hf_name="HSN",
            n_workers=3,
            val_split=0.2,
            task="multilabel",
            classlimit=500,
            eventlimit=5,
            sample_rate=32000,
        ),
        loaders: LoadersConfig = LoadersConfig(),
        transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
        mapper: XCEventMapping = XCEventMapping(),
        weightsampler_column_name: str = "labels",
    ):
        super().__init__(dataset, loaders, transforms, mapper, weightsampler_column_name)
        self.calltype_map = calltype_map
        self.filter_naive = filter_naive
        self.unknown_ebird_code = unknown_ebird_code
        self.num_combined_classes = num_combined_classes
        self.filter_unspecified = filter_unspecified
        self.limit_samples = limit_samples

    @property
    def num_classes(self):
        # This is a arbitrary value, which is only used to setup the backbone
        return 42

    @property
    def num_classes_ebird(self):
        return len(self.ebird_labels)

    @property
    def num_classes_calltype(self):
        return len(self.calltype_labels)
    
    def _process_loaded_multitask_data(self, dataset, decode: bool = True):
        if isinstance(dataset, IterableDataset | IterableDatasetDict):
            log.error("Iterable datasets not supported yet.")
            return

        assert isinstance(dataset, DatasetDict | Dataset)
        dataset = self._ensure_train_test_splits(dataset)
        def add_multilabel_column(example):
            example["ebird_code_multilabel"] = example["ebird_code"]
            example["call_type_multilabel"] = example["short_call_type"]
            example["combined_multilabel"] = example["ebird_code_and_call"]
            return example
        
        dataset = dataset.map(add_multilabel_column)

        ebird_labels = set()
        calltype_labels = set()
        combined_labels = set()

        for split in dataset.keys():
            ebird_labels.update(dataset[split]["ebird_code"])
            calltype_labels.update(dataset[split]["short_call_type"])
            combined_labels.update(dataset[split]["ebird_code_and_call"])

        self.ebird_labels = sorted(list(ebird_labels))
        self.calltype_labels = sorted(list(calltype_labels))
        self.combined_labels = sorted(list(combined_labels))

        if (len(self.combined_labels) != self.num_combined_classes):
            raise ValueError(f"Number of combined classes {len(self.combined_labels)} does not match expected {self.num_combined_classes}.")

        ebird_label_to_id = {lbl: i for i, lbl in enumerate(self.ebird_labels)}
        calltype_label_to_id = {lbl: i for i, lbl in enumerate(self.calltype_labels)}
        combined_label_to_id = {lbl: i for i, lbl in enumerate(self.combined_labels)}

        def label_to_id_fn(batch):
            for i in range(len(batch["ebird_code_multilabel"])):
                batch["ebird_code_multilabel"][i] = ebird_label_to_id[batch["ebird_code_multilabel"][i]]

            for i in range(len(batch["call_type_multilabel"])):
                batch["call_type_multilabel"][i] = calltype_label_to_id[batch["call_type_multilabel"][i]]

            for i in range(len(batch["combined_multilabel"])):
                batch["combined_multilabel"][i] = combined_label_to_id[batch["combined_multilabel"][i]]

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

        # print length of ebird_labels and calltype_labels
        print(f"Number of unique eBird labels: {self.num_classes_ebird}")
        print(f"Number of unique call type labels: {self.num_classes_calltype}")

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
                dataset[split] = dataset[split].map(
                    self._classes_one_hot,
                    fn_kwargs={
                        "label_column_name": "combined_multilabel",
                        "num_classes": self.num_combined_classes
                    },
                    batched=True,
                    batch_size=300,
                    num_proc=self.dataset_config.n_workers,
                    desc=f"One-hot-encoding combined labels for {split}.",
                )

            dataset = dataset.rename_column("ebird_code_multilabel", "labels_ebird")
            dataset = dataset.rename_column("call_type_multilabel", "labels_calltype")
            dataset = dataset.rename_column("combined_multilabel", "labels_combined")

            dataset_test = dataset.pop("test_5s")
            dataset["test"] = dataset_test
        else:
            raise f"{self.dataset_config.task=} is not supported, choose (multilabel, multiclass)"

        for split in ["train", "test"]:
            dataset[split] = dataset[split].select_columns(
                ["filepath", "labels_ebird", "labels_calltype", "labels_combined", "detected_events", "start_time", "end_time"]
            )

        return dataset