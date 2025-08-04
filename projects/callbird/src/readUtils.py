def readLabeledMapping(file_path: str, label: str | None):
    """
    Reads a labeled mapping from a file (including unlabeled data).
    Args:
        file_path (str): Path to the file containing the labeled mapping.
        label (str | None): The label to filter the mapping or None if map of all labels is needed.
    """
    mapping = {}
    current_label = None
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            if line.startswith("@"):
                current_label = line.strip()[1:]
                if label is None:
                    mapping[current_label] = {}
            elif current_label == label or label is None or current_label is None:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if label is None:
                        mapping[current_label][key] = value
                    else:
                        mapping[key] = value
                elif line.strip() == "":
                    continue
                else:
                    print(f"Skipping malformed line: {line.strip()}")
    return mapping
