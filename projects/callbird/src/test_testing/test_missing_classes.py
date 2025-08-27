from os import path
from callbird.src.readUtils import readCommentedList, readLabeledMapping
from birdset.datamodule import BirdSetDataModule
from datasets import load_dataset, DatasetDict, IterableDataset, IterableDatasetDict, Audio, Features, Value, Dataset

def check():
    ebird_labels = set()
    calltype_labels = set()

    for split in dataset.keys():
        ebird_labels.update(dataset[split]["ebird_code"])
        calltype_labels.update(dataset[split]["short_call_type"])

    ebird_labels = sorted(list(ebird_labels))
    calltype_labels = sorted(list(calltype_labels))
    
    # print length of ebird_labels and calltype_labels
    num_classes_ebird = len(ebird_labels)
    num_classes_calltype = len(calltype_labels)

    print(f"Number of unique eBird labels: {num_classes_ebird}")
    print(f"eBird labels: {ebird_labels}")

    print(f"Number of unique call type labels: {num_classes_calltype}")
    print(f"Call type labels: {calltype_labels}")

    expected_values = set(['winwre4', 'comcha', 'gretit1', 'misthr1', 'hawfin', 'coatit2', 'eurrob1', 'blutit', 'grswoo', 'goldcr1', 'yellow2', 'shttre1', 'brambl', 'eurgol', 'eurnut2', 'UNKNOWN', 'dunnoc1', 'sonthr1', 'norlap', 'eurbla', 'martit2', 'wlwwar', 'cowpig1', 'NA', 'eurjay1', 'comchi1', 'cretit2', 'eugwoo2', 'firecr1', 'stodov1', 'eursta', 'tawowl1', 'blackc1', 'eurgre1', 'spofly1', 'gyfwoo1', 'woowar', 'comcuc', 'eurtre1', 'eupfly1', 'miswoo1', 'blawoo1', 'eutdov', 'eurjac', 'wiltit1', 'whtdip1', 'eurbul', 'redcro', 'eursis', 'redwin', 'fieldf', 'carcro1', 'skylar', 'norgos1', 'grewhi1', 'comnig1', 'comrav', 'redjun', 'garwar1', 'trepip', 'eurwoo'])

    missing_codes = expected_values - set(ebird_labels)
    added_codes = set(ebird_labels) - expected_values

    print(f"Missing eBird codes: {missing_codes}")
    print(f"Added eBird codes: {added_codes}")

    # force process exit
    import sys; sys.exit()

# A list of classes not present in the train set.
# blacklist_ebird = readCommentedList("/workspace/projects/callbird/datastats/test/blacklist_ebird.txt")
blacklist_naive = readCommentedList("/workspace/projects/callbird/datastats/test/blacklist_naive.txt")

dataset = load_dataset(
    "csv",
    data_files = "/workspace/oekofor/testset/labels/*.csv",
    features = Features({ # TODO: Add all features available in BirdSet
        "tsn_code": Value("string"),
        "ebird_code": Value("string"),
        "common_name": Value("string"),
        "vocalization_type": Value("string"),
        "start_time": Value("float"),
        "end_time": Value("float"),
        "audio_filename": Value("string"),
    }),
    cache_dir = None,
    num_proc = 1,
    trust_remote_code = True, # While not needed for local datasets, it is kept for consistency
)

# We need to remove None values from the 'ebird_code' column since the pipeline cannot handle them
dataset = dataset.map(lambda x: {"ebird_code": x["ebird_code"] if x["ebird_code"] is not None else ("UNKNOWN" if x["common_name"] == "Bird" else "NA")}) # TODO: Check if NA is an existing code
dataset = dataset.map(lambda x: {"vocalization_type": x["vocalization_type"] if x["vocalization_type"] is not None else "NA"}) # TODO: Check if NA is an existing code

# Load the call type mappings
calltype_mapping = readLabeledMapping("/workspace/projects/callbird/datastats/call_types_list", "test")
dataset = dataset.map(lambda x: {"short_call_type": calltype_mapping.get(x["vocalization_type"], None)}) # Using None to force an error if the vocalization type is not found

# Create naive classes
dataset = dataset.map(lambda x: {"ebird_code_and_call": f"{x['ebird_code']}_{x['short_call_type']}"})

# Filter out entries with eBird codes in the blacklist
# dataset = dataset.filter(lambda x: x["ebird_code_and_call"] not in blacklist_naive)

check()
