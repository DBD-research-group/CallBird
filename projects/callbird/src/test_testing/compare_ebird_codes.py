from datasets import Value, load_dataset, Features
import os

# print(os.getcwd()) #oekofor/testset/labels/SN001_2019_02_22_08_26_35_FeMaAp_morning.csv

# print(load_dataset_builder("csv", data_files="../../oekofor/testset/labels/SN001_2019_02_22_08_26_35_FeMaAp_morning.csv").info.features)

test_dataset = load_dataset("csv", data_files="/workspace/oekofor/testset/labels/*.csv")

test_ebirds = test_dataset["train"].unique("ebird_code")

train_dataset = load_dataset(
    "csv",
    data_files="/workspace/oekofor/trainset/csvlabels/*.csv",
    delimiter=";",
    features=Features(
        {
            "ebird_code": Value("string"),
        }
    )
)

train_ebirds = train_dataset["train"].unique("ebird_code")

not_in_test = set(test_ebirds) - set(train_ebirds)
print("Ebird codes in train but not in test:")
print(not_in_test)