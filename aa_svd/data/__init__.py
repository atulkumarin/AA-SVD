import logging
import os
from typing import Any, Optional, Tuple

from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def create_datasets(
    config: OmegaConf,
    tokenizer: Optional[Any] = None,
) -> Tuple[Dataset, Dataset]:
    """Create train/val datasets based on configuration."""
    dataset_type = config.get('type')
    if dataset_type == 'huggingface':
        return _load_huggingface_dataset(config, tokenizer)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _load_huggingface_dataset(
    config: OmegaConf,
    tokenizer: Optional[Any] = None,
) -> Tuple[Dataset, Dataset]:
    """Load a HuggingFace dataset and cache it to disk as binary shards."""
    block_size = config.get('block_size', 512)
    num_samples = config.get('num_samples', None)
    num_samples_val = config.get('num_samples_val', num_samples)
    sampling = config.get('sampling', 'random')
    seed = config.get('seed', 42)

    data_path = os.path.join(
        config.get('data_path'),
        config.get('name'),
        config.get('subset'),
        tokenizer.name_or_path,
        config.get('task_type'),
    )

    if os.path.exists(data_path):
        try:
            logger.info(f"Loading dataset from {data_path}")
            from .iterable_text_dataset import TextCalibrationDataset
            train_dataset = TextCalibrationDataset(
                data_path, block_size=block_size,
                num_samples=num_samples, sampling=sampling,
                split='train', seed=seed,
            )
            val_dataset = TextCalibrationDataset(
                data_path, block_size=block_size,
                num_samples=num_samples_val, sampling=sampling,
                split='val', seed=seed,
            )
            return train_dataset, val_dataset
        except Exception as e:
            logger.warning(
                f"Failed to load dataset from disk: {e}. "
                "Will download from HuggingFace and preprocess."
            )

    name = config.get('name')
    subset = config.get('subset')
    logger.info(f"Loading subset={subset} for dataset={name} from HuggingFace")

    if name == "c4":
        url = (
            "https://huggingface.co/datasets/allenai/c4/resolve/main"
            "/en/c4-train.00000-of-01024.json.gz"
        )
        dataset = load_dataset("json", data_files=url)
        val_url = (
            "https://huggingface.co/datasets/allenai/c4/resolve/main"
            "/en/c4-validation.00000-of-00008.json.gz"
        )
        dataset["val"] = load_dataset("json", data_files=val_url)["train"]
    else:
        dataset = load_dataset(name, subset, num_proc=8)

    if hasattr(config, 'preprocessing'):
        from .utils import _apply_preprocessing
        dataset = _apply_preprocessing(
            name, dataset, config.preprocessing, tokenizer
        )

    from .utils import save_dataset_to_disk
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    logger.info(f"Saving dataset to {data_path}")
    save_dataset_to_disk(dataset, data_path)

    from .iterable_text_dataset import TextCalibrationDataset
    train_dataset = TextCalibrationDataset(
        data_path, block_size=block_size,
        num_samples=num_samples, sampling=sampling,
        split='train', seed=seed,
    )
    val_dataset = TextCalibrationDataset(
        data_path, block_size=block_size,
        num_samples=num_samples_val, sampling=sampling,
        split='val', seed=seed,
    )
    return train_dataset, val_dataset


def get_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader for the given dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
