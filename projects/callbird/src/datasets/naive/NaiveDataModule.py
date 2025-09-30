from birdset import utils
from birdset.datamodule import BirdSetDataModule
from datasets import DatasetDict, IterableDataset, IterableDatasetDict, Audio, Dataset

from datasets import DatasetDict

from birdset import utils
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.components.event_mapping import XCEventMapping
from birdset.configs import DatasetConfig, LoadersConfig

log = utils.get_pylogger(__name__)

class NaiveDataModule(BirdSetDataModule):

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
        return self.num_combined_classes
    
    def _process_loaded_multitask_data(self, dataset, decode: bool = True):
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
            # Add alias labels_combined for comparability with MultiDataModule
            dataset = dataset.map(
                lambda ex: {**ex, "labels_combined": ex["labels_ebird"]},
                batched=False,
                desc="Alias combined labels",
            )

            dataset_test = dataset.pop("test_5s")
            dataset["test"] = dataset_test
        else:
            raise f"{self.dataset_config.task=} is not supported, choose (multilabel, multiclass)"

        for split in ["train", "test"]:
            dataset[split] = dataset[split].select_columns(
                ["filepath", "labels_ebird", "labels_combined", "detected_events", "start_time", "end_time"]
            )

        return dataset