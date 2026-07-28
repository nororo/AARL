"""
PyTorch Lightning DataModule

Build train/val/test DataLoader for CLIP learning.
If text_type has multiple types, use a custom batch sampler.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from .config import MultiAccountCLIPConfig
from .datasets import (
    MixedTextTypeBatchSampler,
    MultiAccountDataset,
    TextTypeBatchSampler,
)

logger = logging.getLogger(__name__)

TRAIN_SPLIT_RATIO = 0.9


class MultiAccountDataModule(pl.LightningDataModule):
    """CLIP学習用 DataModule

    Args:
        companies_data: list of company data returned by load_real_dataset
        texts: text for each sample
        tokenizer: tokenizer for text
        config: configuration
        text_types: text_type for each sample (None means normal shuffling)
        docids: docid for each sample (used to avoid docid duplication within a batch)
        use_text_type_batching: use text_type based batch sampler
        use_mixed_batching: True for MixedTextTypeBatchSampler (joint), False for TextTypeBatchSampler (alternating)
        pretraining_split_indices: (train_indices, val_indices) used in pretraining. If specified, reuse the same split.
    """

    def __init__(
        self,
        companies_data: List[Dict],
        texts: List[str],
        tokenizer: AutoTokenizer,
        config: MultiAccountCLIPConfig,
        text_types: Optional[List[str]] = None,
        docids: Optional[List[str]] = None,
        use_text_type_batching: bool = True,
        use_mixed_batching: bool = True,
        pretraining_split_indices: Optional[Tuple[List[int], List[int]]] = None,
    ):
        super().__init__()
        self.companies_data = companies_data
        self.texts = texts
        self.tokenizer = tokenizer
        self.config = config
        self.text_types = text_types
        self.docids = docids
        self.use_text_type_batching = use_text_type_batching
        self.use_mixed_batching = use_mixed_batching
        self.pretraining_split_indices = pretraining_split_indices

        self.dataset: Optional[MultiAccountDataset] = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_text_types: Optional[List[str]] = None
        self.val_text_types: Optional[List[str]] = None
        self.train_docids: Optional[List[str]] = None
        self.val_docids: Optional[List[str]] = None

    def setup(self, stage: Optional[str] = None):
        """Build dataset and train/val split
        """
        if self.dataset is None:
            self.dataset = MultiAccountDataset(
                self.companies_data,
                self.texts,
                self.tokenizer,
                self.config.max_length,
                self.text_types,
                self.docids,
            )

        if self.train_dataset is not None:
            return

        total_size = len(self.dataset)

        if self.pretraining_split_indices is not None:
            # Reuse the same train/val split for pretraining
            train_indices, val_indices = self.pretraining_split_indices
            logger.info(
                f"Reuse train/val split for pretraining: "
                f"Train={len(train_indices)}, Validation={len(val_indices)}"
            )
        else:
            indices = list(range(total_size))
            np.random.shuffle(indices)
            train_size = int(TRAIN_SPLIT_RATIO * total_size)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = self.val_dataset

        if self.text_types is not None:
            self.train_text_types = [self.text_types[i] for i in train_indices]
            self.val_text_types = [self.text_types[i] for i in val_indices]

        if self.docids is not None:
            self.train_docids = [self.docids[i] for i in train_indices]
            self.val_docids = [self.docids[i] for i in val_indices]

        logger.info(
            f"Data split: Train={len(self.train_dataset)}, "
            f"Validation={len(self.val_dataset)}, Test={len(self.test_dataset)}"
        )

        if self.use_text_type_batching and self.train_text_types is not None:
            train_type_counts = pd.Series(self.train_text_types).value_counts()
            val_type_counts = pd.Series(self.val_text_types).value_counts()
            logger.info(f"Text type distribution in training data: {train_type_counts.to_dict()}")
            logger.info(f"Text type distribution in validation data: {val_type_counts.to_dict()}")

            if len(train_type_counts) == 1:
                logger.warning(
                    f"Text type in training data is only one type ({train_type_counts.index[0]})."
                    "Text type based batch sampling is disabled."
                )

    def _make_dataloader(
        self,
        dataset,
        text_types: Optional[List[str]],
        docids: Optional[List[str]],
        shuffle: bool,
    ) -> DataLoader:
        """Create DataLoader with custom sampler or normal DataLoader depending on the presence of text_type."""
        use_custom_sampler = (
            self.use_text_type_batching
            and text_types is not None
            and docids is not None
            and len(set(text_types)) > 1  # only valid if text_type has multiple types
        )

        if not use_custom_sampler:
            if self.use_text_type_batching and shuffle:
                logger.info(
                    "Use normal random shuffling because text_type is only one type or not available."
                )
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                collate_fn=self.dataset.collate_fn,
            )

        if self.use_mixed_batching:
            # 真のjoint optimization: 複数text_typeを混在させたバッチ
            if shuffle:
                logger.info("MixedTextTypeBatchSamplerを使用します（真のjoint optimization）")
            sampler_cls = MixedTextTypeBatchSampler
        else:
            # Alternating optimization: 単一text_typeのバッチ
            if shuffle:
                logger.info("TextTypeBatchSamplerを使用します（alternating optimization）")
            sampler_cls = TextTypeBatchSampler

        batch_sampler = sampler_cls(
            text_types=text_types,
            docids=docids,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.config.num_workers,
            collate_fn=self.dataset.collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(
            self.train_dataset, self.train_text_types, self.train_docids, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(
            self.val_dataset, self.val_text_types, self.val_docids, shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        # テストデータは検証データと同じ
        return self._make_dataloader(
            self.test_dataset, self.val_text_types, self.val_docids, shuffle=False
        )
