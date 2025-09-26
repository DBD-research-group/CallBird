from callbird.src.readUtils import readCommentedList, readLabeledMapping
from datasets import concatenate_datasets, load_dataset, Features, Value
from os import path

def load_train_dataset(
        call_type_mapping_file: str,
        cache_dir: str | None = None
):
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
        cache_dir = cache_dir,
        num_proc = 1,
        trust_remote_code = True, # While not needed for local datasets, it is kept for consistency
    )

    # We need to remove None values from the 'ebird_code' column since the pipeline cannot handle them
    dataset = dataset.map(lambda x: {"ebird_code": x["ebird_code"] if x["ebird_code"] is not None else "NA"})
    dataset = dataset.map(lambda x: {"call_type": x["call_type"] if x["call_type"] is not None else "NA"})

    # A list of classes not present in the train set (this is empty, but it is kept for consistency).
    blacklist_ebird = readCommentedList("/workspace/projects/callbird/datastats/train/blacklist_ebird.txt")
    dataset = dataset.filter(lambda x: x["ebird_code"] not in blacklist_ebird)

    # Limit the number of "NA" ebird_code entries to 5000
    # na_dataset = dataset.filter(lambda x: x["ebird_code"] == "NA")
    # other_dataset = dataset.filter(lambda x: x["ebird_code"] != "NA")
    # na_subset = na_dataset['train'].shuffle(seed=42).select(range(min(5000, len(na_dataset['train']))))
    # dataset['train'] = concatenate_datasets([other_dataset['train'], na_subset])

    # Load the call type mappings
    calltype_mapping = readLabeledMapping(call_type_mapping_file, "train")
    dataset = dataset.map(lambda x: {"short_call_type": calltype_mapping.get(x["call_type"], None)}) # Using None to force an error if the call type is not found

    # Create naive classes
    dataset = dataset.map(lambda x: { "ebird_code_and_call": f"{x['ebird_code']}_{x['short_call_type']}" })

    blacklist_naive = readCommentedList("/workspace/projects/callbird/datastats/train/blacklist_naive.txt")
    dataset = dataset.filter(lambda x: x["ebird_code_and_call"] not in blacklist_naive)

    #### TODO: Reduce samples etc.

    dataset = dataset.rename_column("start_sample [s]", "start_time")
    dataset = dataset.rename_column("end_sample [s]", "end_time")
    dataset = dataset.rename_column("actual_filename", "filepath")

    # Setting absolute paths for the audio files
    def update_filepath(example):
        example["filepath"] = f"/workspace/oekofor/dataset/{example['filepath']}.flac"

        if not path.exists(example["filepath"]):
            example["filepath"] = example["filepath"].replace(".flac", ".wav")

        return example

    dataset = dataset.map(update_filepath)

    # Create detected_events from start_time and end_time
    dataset = dataset.map(lambda x: {"detected_events": [x["start_time"], x["end_time"]]})
    
    return dataset