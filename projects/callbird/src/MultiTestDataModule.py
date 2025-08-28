from os import path
from callbird.src.MultiDataModule import MultiDataModule
from callbird.src.ensure_torch_safe_globals import ensure_torch_safe_globals
from callbird.src.readUtils import readCommentedList, readLabeledMapping
from birdset.datamodule import BirdSetDataModule
from datasets import load_dataset, DatasetDict, Features, Value

# I don't know why this is needed, since it works for others without, but due to the limited time, this is here to stay
ensure_torch_safe_globals()

class MultiTestDataModule(MultiDataModule):
    """
    A class to handle the test dataset for Oekofor.
    This class is designed to process the dataset, ensuring it meets the requirements
    of the base class, including renaming columns, constructing file paths, and adding
    necessary columns for event detection.
    """
    
    def _load_data(self, decode: bool = True) -> DatasetDict:
        # A list of classes not present in the train set.
        blacklist_ebird = readCommentedList("/workspace/projects/callbird/datastats/test/blacklist_ebird.txt")
        # blacklist_naive = readCommentedList("/workspace/projects/callbird/datastats/test/blacklist_naive.txt")

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
            cache_dir = self.dataset_config.data_dir,
            num_proc = 1,
            trust_remote_code = True, # While not needed for local datasets, it is kept for consistency
        )

        # We need to remove None values from the 'ebird_code' column since the pipeline cannot handle them
        dataset = dataset.map(lambda x: {"ebird_code": x["ebird_code"] if x["ebird_code"] is not None else ("UNKNOWN" if x["common_name"] == "Bird" else "NA")}) # TODO: Check if NA is an existing code
        dataset = dataset.map(lambda x: {"vocalization_type": x["vocalization_type"] if x["vocalization_type"] is not None else "NA"}) # TODO: Check if NA is an existing code

        # Load the call type mappings
        calltype_mapping = readLabeledMapping("/workspace/projects/callbird/datastats/call_types_list", "test")
        dataset = dataset.map(lambda x: {"short_call_type": calltype_mapping.get(x["vocalization_type"], None)}) # Using None to force an error if the vocalization type is not found

        dataset = dataset.filter(lambda x: x["ebird_code"] not in blacklist_ebird)

        # Rename 'audio_filename' to 'filepath' to match what the base class expects
        dataset = dataset.rename_column("audio_filename", "filepath")

        # Setting absolute paths for the audio files
        def update_filepath(example):
            example["filepath"] = f"/workspace/oekofor/testset/audio_files/{example['filepath']}.flac"

            if not path.exists(example["filepath"]):
                example["filepath"] = example["filepath"].replace(".flac", ".wav")

            return example

        dataset = dataset.map(update_filepath)

        dataset = super()._process_loaded_multitask_data(dataset, decode)

        return dataset