from callbird.src.datasets.multi.MultiDataModule import MultiDataModule
from callbird.src.datasets.load_test_dataset import load_test_dataset
from callbird.src.ensure_torch_safe_globals import ensure_torch_safe_globals
from datasets import DatasetDict

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
        dataset = load_test_dataset(
            call_type_mapping_file=self.calltype_map,
            cache_dir=self.dataset_config.data_dir,
            filter_naive=self.filter_naive,
            unknown_ebird_code=self.unknown_ebird_code
        )

        dataset = super()._process_loaded_multitask_data(dataset, decode)

        return dataset