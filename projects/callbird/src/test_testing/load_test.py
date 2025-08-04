from datasets import Value, load_dataset, Features
import os

# print(os.getcwd()) #oekofor/testset/labels/SN001_2019_02_22_08_26_35_FeMaAp_morning.csv

# print(load_dataset_builder("csv", data_files="../../oekofor/testset/labels/SN001_2019_02_22_08_26_35_FeMaAp_morning.csv").info.features)

# dataset = load_dataset("csv", data_files="/workspace/oekofor/testset/labels/*.csv")

# print(len(dataset["train"].unique("ebird_code")))

dataset = load_dataset(
    "csv",
    data_files="/workspace/oekofor/trainset/csvlabels/*.csv",
    delimiter=";",
    features=Features(
        {
            "ebird_code": Value("string"),
            "call_type": Value("string"),
        }
    )
)

print(len(dataset["train"].unique("call_type")))

# list all call_type
print(dataset["train"].unique("call_type"))