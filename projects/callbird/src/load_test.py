import os
from birdset.datamodule import BirdSetDataModule
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Audio, load_dataset, load_from_disk, load_dataset_builder
import os

# print(os.getcwd()) #oekofor/testset/labels/SN001_2019_02_22_08_26_35_FeMaAp_morning.csv

# print(load_dataset_builder("csv", data_files="../../oekofor/testset/labels/SN001_2019_02_22_08_26_35_FeMaAp_morning.csv").info.features)

dataset = load_dataset("csv", data_files="../../oekofor/testset/labels/*.csv")

print(len(dataset["train"].unique("ebird_code")))