import pickle
import sys
from collections import defaultdict
import zipfile
import io

class Placeholder:
    """A dummy class to stand in for unpickled types."""
    def __getattr__(self, name):
        # Return another placeholder for attribute access.
        return Placeholder()
    def __setstate__(self, state):
        # Some objects might require this during unpickling.
        pass

class TypeFinder(pickle.Unpickler):
    """A custom unpickler to find all class types in a pickle stream."""
    def __init__(self, file, found_types):
        super().__init__(file)
        self.found_types = found_types

    def find_class(self, module, name):
        """This method is called by the unpickler to find a class."""
        # Ignore built-in types which are generally safe
        if module not in ["builtins", "__builtin__"]:
             self.found_types[module].add(name)
        
        # Return a placeholder class to allow the unpickler to continue
        # processing the object graph, revealing more types.
        return Placeholder

def process_pickle_stream(stream, found_types):
    """Runs the TypeFinder unpickler on a file-like stream."""
    unpickler = TypeFinder(stream, found_types)
    while True:
        try:
            # Load objects one by one until EOF
            unpickler.load()
        except EOFError:
            # Reached end of file
            break
        except Exception:
            # An error is expected because we are not actually loading data.
            # We can continue to try and read from the stream, as there might
            # be more independent objects pickled.
            continue


def inspect_checkpoint_types(checkpoint_path):
    """
    Inspects a PyTorch checkpoint file to find all global types
    without fully loading it.
    """
    found_types = defaultdict(set)
    print(f"Inspecting checkpoint: {checkpoint_path}\n")
    
    try:
        # First, try to open as a zip archive (common for .ckpt files)
        with zipfile.ZipFile(checkpoint_path, 'r') as zf:
            # Common locations for pickled objects in PyTorch/Lightning checkpoints
            pickle_files = [
                name for name in zf.namelist() 
                if name.endswith(('.pkl', '/data.pkl', 'hparams.pkl', 'hyper_parameters.pkl'))
            ]
            if not pickle_files:
                print("Error: Could not find any .pkl files inside the zip archive.")
                return

            print(f"Found pickle data in: {', '.join(pickle_files)}")
            for pickle_file in pickle_files:
                 with zf.open(pickle_file, 'r') as f:
                    process_pickle_stream(io.BytesIO(f.read()), found_types)

    except zipfile.BadZipFile:
        # If it's not a zip file, treat it as a raw pickle file
        print("File is not a zip archive, treating as raw pickle data.")
        with open(checkpoint_path, 'rb') as f:
            process_pickle_stream(f, found_types)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


    if not found_types:
        print("\nNo non-built-in types were found. The file might be empty, corrupted, or use a format this script doesn't handle.")
        return

    print("\nFound the following types that may need to be allowed:")
    import_statements = []
    class_names = []
    for module, names in sorted(found_types.items()):
        for name in sorted(names):
            # Handle special cases like omegaconf where the import path
            # doesn't directly match the pickled module path.
            if module.startswith("omegaconf"):
                print(f"  - from omegaconf import {name}")
                import_statements.append(f"from omegaconf import {name}")
            else:
                print(f"  - from {module} import {name}")
                import_statements.append(f"from {module} import {name}")
            class_names.append(name)
    
    print("\n--- Suggested Code ---")
    print("\n# 1. Add these imports to your script:")
    print("\n".join(sorted(import_statements)))
    
    print("\n# 2. Add these classes to the safe globals list:")
    print(f"torch.serialization.add_safe_globals([{', '.join(sorted(class_names))}])")
    print("----------------------")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_checkpoint.ckpt>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    inspect_checkpoint_types(checkpoint_path)