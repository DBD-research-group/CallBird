from callbird.src.readUtils import readCommentedList, readLabeledMapping
from datasets import load_dataset, Features, Value
from os import path

def load_test_dataset(
        call_type_mapping_file: str,
        cache_dir: str | None,
        filter_naive: str | None,
        unknown_ebird_code: str,
        filter_unspecified: bool,
):
    dataset = load_dataset(
        "csv",
        data_files = "/workspace/oekofor/testset/labels/*.csv",
        features = Features({ # TODO: Add all features available in BirdSet
            "ebird_code": Value("string"),
            "common_name": Value("string"),
            "vocalization_type": Value("string"),
            "start_time": Value("float"),
            "end_time": Value("float"),
            "audio_filename": Value("string"),
        }),
        cache_dir = cache_dir,
        num_proc = 1,
        trust_remote_code = True, # While not needed for local datasets, it is kept for consistency
    )

    # We need to remove None values from the 'ebird_code' column since the pipeline cannot handle them
    dataset = dataset.map(lambda x: {"ebird_code": x["ebird_code"] if x["ebird_code"] is not None else (unknown_ebird_code if x["common_name"] == "Bird" else "NA")}) # TODO: Check if NA is an existing code
    dataset = dataset.map(lambda x: {"vocalization_type": x["vocalization_type"] if x["vocalization_type"] is not None else "NA"}) # TODO: Check if NA is an existing code

    # If the ebird_code is NA, we also set the vocalization_type to NA
    dataset = dataset.map(lambda x: {"vocalization_type": "NA" if x["ebird_code"] == "NA" and x["common_name"] != "Bird" else x["vocalization_type"]})

    # A list of classes not present in the train set.
    blacklist_ebird = readCommentedList("/workspace/projects/callbird/datastats/test/blacklist_ebird.txt")
    dataset = dataset.filter(lambda x: x["ebird_code"] not in blacklist_ebird)

    # Load the call type mappings
    calltype_mapping = readLabeledMapping(call_type_mapping_file, "test")
    dataset = dataset.map(lambda x: {"short_call_type": calltype_mapping.get(x["vocalization_type"], None)}) # Using None to force an error if the vocalization type is not found

    # Create naive classes
    dataset = dataset.map(lambda x: { "ebird_code_and_call": f"{x['ebird_code']}_{x['short_call_type']}" })

    if filter_naive != None:
        blacklist_naive = readCommentedList(filter_naive)
        dataset = dataset.filter(lambda x: x["ebird_code_and_call"] not in blacklist_naive)

    #### TODO: Reduce samples etc.

    if filter_unspecified:
        dataset = dataset.filter(lambda x: x["common_name"] != "Bird")

    # Rename 'audio_filename' to 'filepath' to match what the base class expects
    dataset = dataset.rename_column("audio_filename", "filepath")

    # Setting absolute paths for the audio files
    def update_filepath(example):
        example["filepath"] = f"/workspace/oekofor/testset/audio_files/{example['filepath']}.flac"

        if not path.exists(example["filepath"]):
            example["filepath"] = example["filepath"].replace(".flac", ".wav")

        return example

    dataset = dataset.map(update_filepath)

    # Create detected_events from start_time and end_time
    dataset = dataset.map(lambda x: {"detected_events": [x["start_time"], x["end_time"]]})

    return dataset