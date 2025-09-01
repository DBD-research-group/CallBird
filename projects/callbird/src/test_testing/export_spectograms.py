import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import torch
import librosa.display
import numpy as np
import os
import sys

# We remove the default @hydra.main decorator to load the config manually
def generate_spectrograms(cfg: DictConfig) -> None:
    """
    Generates and saves spectrograms using a specific, resolved hydra config.
    """
    print("Configuration used:")
    print(OmegaConf.to_yaml(cfg.datamodule.transforms))

    print("\nInstantiating datamodule from saved config...")
    # We need to set _recursive_=False because the saved config is already resolved.
    datamodule = hydra.utils.instantiate(cfg.datamodule, _recursive_=True)
    
    # Call prepare_data() before setup() to ensure data is downloaded/processed
    # and the disk_save_path is set.
    print("Preparing data...")
    datamodule.prepare_data()

    print("Setting up datamodule...")
    datamodule.setup(stage='fit')

    dataloader = datamodule.train_dataloader()
    dataset = dataloader.dataset

    # --- DEBUG: Inspect the first item ---
    first_item = dataset[0]
    print("\nKeys in a single dataset item:", first_item.keys())
    # You can now exit to check the keys before running the full loop
    # import sys; sys.exit()
    # --- END DEBUG ---

    output_dir = "/workspace/projects/callbird/datastats/spectrogram_exports_from_config"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nExporting spectrograms to '{output_dir}/'")

    # Get spectrogram parameters from the config for accurate plotting
    transforms_cfg = cfg.datamodule.transforms
    spec_params = transforms_cfg.get('spectrogram_transformer', {})
    mel_spec_params = spec_params.get('params', {})
    
    sample_rate = transforms_cfg.get('sample_rate', 32000)
    n_mels = mel_spec_params.get('n_mels', 128)
    n_fft = mel_spec_params.get('n_fft', 1024)
    hop_length = mel_spec_params.get('hop_length', 320)

    num_samples_to_export = 5
    exported_count = 0
    for i, item in enumerate(dataset):
        if exported_count >= num_samples_to_export:
            break
        
        # The rest of the logic remains the same...
        if not isinstance(item, dict) or 'input_values' not in item:
            print(f"Skipping sample {i}, 'input_values' key not found.")
            continue

        spectrogram = item['input_values'].cpu()
        ebird_code_multilabel = item['labels_ebird'].cpu()
        calltype_multilabel = item["labels_calltype"].cpu()

        # --- Convert one-hot vectors back to string labels ---
        
        # Get the index-to-label mapping from the datamodule
        ebird_idx_to_label = datamodule.ebird_labels
        calltype_idx_to_label = datamodule.calltype_labels

        # Find the indices where the one-hot vector is 1
        ebird_indices = torch.nonzero(ebird_code_multilabel, as_tuple=True)[0]
        calltype_indices = torch.nonzero(calltype_multilabel, as_tuple=True)[0]

        # Convert indices to a comma-separated string of labels
        ebird_code = ", ".join([ebird_idx_to_label[i] for i in ebird_indices]) if len(ebird_indices) > 0 else "N/A"
        calltype = ", ".join([calltype_idx_to_label[i] for i in calltype_indices]) if len(calltype_indices) > 0 else "N/A"
        
        # --- End of conversion ---
        

        plot_path = os.path.join(output_dir, f"sample_{i}_spectrogram.png")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        if spectrogram.ndim == 3:
            spectrogram = spectrogram.squeeze(0)
        
        spectrogram_np = spectrogram.numpy()

        img = librosa.display.specshow(spectrogram_np, sr=sample_rate, hop_length=hop_length,
                                       x_axis='time', y_axis='mel', ax=ax, fmax=sample_rate / 2)
        fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Decibels')
        ax.set_title(f"Sample {i} Spectrogram ({ebird_code} - {calltype})")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        
        print(f"Exported sample {i}: {plot_path}")
        exported_count += 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_spectrograms.py <path_to_hydra_run_config>")
        print("Example: python generate_spectrograms.py outputs/2025-08-31/14-30-00/.hydra/config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    print(f"Loading configuration from: {config_path}")
    
    # Load the specific configuration file
    cfg = OmegaConf.load(config_path)
    
    generate_spectrograms(cfg)