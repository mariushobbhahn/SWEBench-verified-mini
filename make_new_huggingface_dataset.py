from pathlib import Path
import json
import datasets
from typing import List

def load_original_dataset() -> datasets.DatasetDict:
    """
    Load the original SWE-bench dataset.
    
    Returns:
    - Original HuggingFace dataset
    """
    return datasets.load_dataset("princeton-nlp/SWE-bench_Verified")

def load_sample_ids(path: Path) -> List[str]:
    """
    Load sample IDs from a JSON file.
    
    Parameters:
    - path: Path to the JSON file containing sample IDs
    
    Returns:
    - List of sample IDs
    """
    with path.open() as f:
        sample_ids = json.load(f)
    
    # Validate IDs
    n_ids = len(sample_ids)
    n_unique_ids = len(set(sample_ids))
    print(f"Number of IDs: {n_ids}")
    print(f"Number of unique IDs: {n_unique_ids}")
    
    if n_ids != n_unique_ids:
        print("Warning: Duplicate IDs found!")
    
    return sample_ids

def filter_dataset_by_ids(dataset: datasets.DatasetDict, ids: List[str]) -> datasets.DatasetDict:
    """
    Filter dataset to only include rows with matching IDs.
    
    Parameters:
    - dataset: Original HuggingFace dataset
    - ids: List of IDs to keep
    
    Returns:
    - Filtered dataset
    """
    def is_in_ids(example):
        return example['instance_id'] in ids

    return dataset.filter(is_in_ids)

def update_dataset_name(dataset: datasets.DatasetDict, new_name: str) -> None:
    """
    Update the dataset name in the dataset object.
    
    Parameters:
    - dataset: Dataset to update
    - new_name: New name for the dataset
    """
    for split in dataset.keys():
        dataset[split].info.builder_name = new_name

def dataset_glimpse(dataset: datasets.DatasetDict, num_rows: int = 5) -> None:
    """
    Print a glimpse of the dataset and metadata.
    
    Parameters:
    - dataset: Dataset to examine
    - num_rows: Number of rows to display
    """
    print("\nDataset Overview:")
    print(f"Number of rows: {dataset.num_rows}")
    
    for split, data in dataset.items():
        print(f"\nGlimpse of '{split}' split ({len(data)} rows):")
        print(data.to_pandas().head(num_rows))

def main() -> None:
    """Create and save a new filtered HuggingFace dataset."""
    # Load original dataset
    print("Loading original dataset...")
    dataset = load_original_dataset()
    
    # Load and validate sample IDs
    print("\nLoading sample IDs...")
    sample_ids = load_sample_ids(Path("data/subsets/size_optimized_sample_ids.json"))
    
    # Filter dataset
    print("\nFiltering dataset...")
    filtered_dataset = filter_dataset_by_ids(dataset, sample_ids)
    print(f"Filtered dataset size: {filtered_dataset.num_rows} rows")
    
    # Update dataset name
    update_dataset_name(filtered_dataset, "swe-bench-verified-mini")
    
    # Show dataset preview
    dataset_glimpse(filtered_dataset)
    
    # Save locally
    output_path = Path("data/filtered_huggingface_dataset")
    print(f"\nSaving dataset to {output_path}")
    filtered_dataset.save_to_disk(output_path)
    
    # Push to HuggingFace
    print("\nPushing dataset to HuggingFace...")
    filtered_dataset.push_to_hub(
        "MariusHobbhahn/swe-bench-verified-mini",
        #private=True
    )
    print("Done!")

if __name__ == "__main__":
    main()

