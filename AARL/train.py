#!/usr/bin/env python3
"""
Training script for Multi-Account Financial CLIP.

Two-stage training:
  step1 pretraining: Account co-occurrence patterns
                     (current-period MLM, change-class prediction, next-period prediction)
  step2 CLIP training: Align company and text representations

Usage examples:

1. Use a YAML config file (recommended):
   python train_multi_account_clip_lightning_refactored.py --config config_base.yaml

   # EDGAR dataset (column names are remapped by load_real_dataset when is_edgar is set)
   python train_multi_account_clip_lightning_refactored.py --config config_edgar.yaml --is_edgar

   # Override YAML settings with command-line arguments
   python train_multi_account_clip_lightning_refactored.py --config config_base.yaml --batch_size 32 --max_epochs 200

2. Command-line arguments only:
   python train_multi_account_clip_lightning_refactored.py \
       --model_dir ./results/multi_account_clip/ \
       --output_data_dir ./results/multi_account_clip/ \
       --data_dir ./data/afm/ \
       --batch_size 16 --max_epochs 100 --learning_rate 2e-5 \
       --accelerator gpu --use_text_type_loss
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow importing multi_account_clip_common from the same directory as this script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from multi_account_clip_common import (
    AccountPretrainer,
    CustomProgressBar,
    LightningMultiAccountFinancialCLIP,
    MaskedAccountDataset,
    MultiAccountCLIPConfig,
    MultiAccountDataModule,
    evaluate_model,
    load_pretrained_account_encoder,
    load_real_dataset,
    parse_args,
    setup_trainer,
)

warnings.filterwarnings("ignore")

# Show detailed CUDA errors (for debugging). To prefer speed, override with
# CUDA_LAUNCH_BLOCKING=0 in the environment.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
torch.set_float32_matmul_precision("medium")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Max tokenized length for account names during pretraining
PRETRAINING_ACCOUNT_NAME_MAX_LENGTH = 128

# Train/val split ratio for pretraining (reused for CLIP training)
PRETRAINING_TRAIN_SPLIT_RATIO = 0.9

# Number of sample embeddings to save after training
NUM_SAMPLE_EMBEDDINGS = 10


# =============================================================================
# Config construction
# =============================================================================

def _parse_text_type_loss_weights(raw) -> Optional[Dict[str, float]]:
    """Normalize per-text_type manual weights to a dict (YAML is dict, CLI is JSON string)."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        weights = json.loads(raw)
        logger.info(f"Loaded manual weight settings: {weights}")
        return weights
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for manual weight settings: {e}")
        logger.warning("Ignoring manual weight settings.")
        return None


def build_config(args) -> MultiAccountCLIPConfig:
    """Build MultiAccountCLIPConfig from parsed arguments.

    Argument names map 1:1 to config field names, so adding a field
    does not require changes here.
    """
    field_names = set(MultiAccountCLIPConfig.__dataclass_fields__)
    values = {key: value for key, value in vars(args).items() if key in field_names}

    # Handle fields whose argument names do not match the config names
    values["enable_pretraining"] = args.enable_pretraining and not args.disable_pretraining
    values["text_type_loss_weights"] = _parse_text_type_loss_weights(args.text_type_loss_weights)

    return MultiAccountCLIPConfig(**values)


def log_config(config: MultiAccountCLIPConfig) -> None:
    """Log a summary of the training configuration."""
    logger.info("Model settings:")
    logger.info(f"   Account Encoder: {config.account_model}")
    logger.info(f"   Text Encoder   : {config.text_model}")
    logger.info(f"   Change classes : {list(config.change_classes)}")

    if config.use_graph_attention:
        logger.info("   Aggregation: Graph Attention")
        logger.info(f"   - GAT Heads : {config.graph_attention_num_heads}")
        logger.info(f"   - GAT Layers: {config.graph_attention_num_layers}")
        logger.info(f"   - Dropout   : {config.graph_attention_dropout}")
    else:
        logger.info("   Aggregation: Company Attention + mean pooling")

    if config.use_text_lora or config.use_account_lora:
        logger.info("   LoRA settings:")
        logger.info(f"   - Applied to: "
                    f"{'Text ' if config.use_text_lora else ''}"
                    f"{'Account' if config.use_account_lora else ''}")
        logger.info(f"   - Rank      : {config.lora_r}")
        logger.info(f"   - Alpha     : {config.lora_alpha}")
        logger.info(f"   - Targets   : {config.lora_target_modules or 'auto-detect'}")

    logger.info(f"   Output dim    : {config.output_dim}")
    logger.info(f"   Temperature   : {config.temperature}")

    logger.info("\nTraining settings:")
    logger.info(f"   Precision     : {config.precision}")
    logger.info(f"   Batch size    : {config.batch_size}")
    logger.info(f"   Learning rate : {config.learning_rate}")
    logger.info(f"   Max epochs    : {config.max_epochs}")
    logger.info(f"   Accelerator   : {config.accelerator}")

    if config.use_text_type_loss:
        logger.info("\nPer-text_type loss: enabled")
        logger.info(f"   Weighting mode: {config.text_type_loss_weight_mode}")
        if config.text_type_loss_weight_mode == "manual" and config.text_type_loss_weights:
            logger.info(f"   Manual weights: {config.text_type_loss_weights}")
    else:
        logger.info("\nPer-text_type loss: disabled")

    if config.use_joint_pretraining_loss and not config.enable_pretraining:
        logger.info("\nJoint Optimization (CLIP + pretraining loss): enabled")
        logger.info(f"   MLM loss weight          : {config.joint_mlm_loss_weight}")
        logger.info(f"   Change-class loss weight : {config.joint_change_class_loss_weight}")
        if config.use_joint_next_period_loss:
            logger.info("\n   Next-period loss: enabled")
            logger.info(f"   Next MLM loss weight          : {config.joint_next_mlm_loss_weight}")
            logger.info(f"   Next change-class loss weight : {config.joint_next_change_class_loss_weight}")
            logger.info(f"   Next-period mask rate         : {config.next_period_mask_probability:.0%}")
        else:
            logger.info("\n   Next-period loss: disabled")
        logger.info("   Pretraining phase will be skipped (jointly optimized during CLIP training)")
    elif config.use_joint_pretraining_loss and config.enable_pretraining:
        logger.info("\nJoint Optimization: disabled (pretraining runs separately)")
        logger.info("   Optimize MLM / change-class loss during pretraining")
        logger.info("   Use CLIP loss only during CLIP training")
    else:
        logger.info("\nJoint Optimization: disabled")

    logger.info("\nBatching strategy:")
    if config.use_text_type_loss:
        logger.info("   Round-Robin Alternating Optimization + Gradient Accumulation")
        logger.info("   -> Each batch contains a single text_type only (TextTypeBatchSampler)")
        logger.info("   -> Take one batch from each text_type in round-robin order")
        logger.info(f"   -> Accumulate gradients over {config.accumulate_grad_batches} batches before update")
        if config.accumulate_grad_batches > 1:
            logger.info(
                f"   Effective batch size: {config.batch_size} x "
                f"{config.accumulate_grad_batches} = "
                f"{config.batch_size * config.accumulate_grad_batches}"
            )
    else:
        logger.info("   Standard batching: no per-text_type loss")
        if config.accumulate_grad_batches > 1:
            logger.info(f"   -> Accumulate gradients over {config.accumulate_grad_batches} batches")

    if config.is_edgar:
        logger.info("\nDataset: EDGAR (column rename cik_year_next -> docID_next applied)")


# =============================================================================
# Persist train/val split
# =============================================================================

def _split_indices_path(config: MultiAccountCLIPConfig) -> str:
    return os.path.join(config.model_dir, "split_indices.json")


def load_split_indices(config: MultiAccountCLIPConfig) -> Optional[Tuple[List[int], List[int]]]:
    """Load a saved train/val split index (or None if missing).

    Needed when resuming pretraining mid-run, or when pretraining and CLIP
    training should share the same split.
    """
    path = _split_indices_path(config)
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        split_data = json.load(f)

    indices = (split_data["train_indices"], split_data["val_indices"])
    logger.info(f"Loaded saved train/val split indices: {path}")
    logger.info(f"  train={len(indices[0])}, val={len(indices[1])}")
    return indices


def create_split_indices(
    config: MultiAccountCLIPConfig, dataset_size: int
) -> Tuple[List[int], List[int]]:
    """Randomly create a train/val split and save it to a file."""
    all_indices = np.random.permutation(dataset_size).tolist()
    train_size = int(PRETRAINING_TRAIN_SPLIT_RATIO * dataset_size)
    train_indices, val_indices = all_indices[:train_size], all_indices[train_size:]

    os.makedirs(config.model_dir, exist_ok=True)
    path = _split_indices_path(config)
    with open(path, "w") as f:
        json.dump({"train_indices": train_indices, "val_indices": val_indices}, f)
    logger.info(f"Saved train/val split indices: {path}")

    return train_indices, val_indices


# =============================================================================
# step1: Pretraining
# =============================================================================

def run_pretraining(
    config: MultiAccountCLIPConfig,
    companies_data: List[Dict],
    split_indices: Optional[Tuple[List[int], List[int]]],
) -> Tuple[torch.nn.Module, float, Tuple[List[int], List[int]]]:
    """Run account-encoder pretraining (step1).

    Returns:
        account_encoder: Pretrained encoder
        elapsed: Elapsed time in seconds
        split_indices: Train/val split used (passed on to CLIP training)
    """
    logger.info("Starting account-encoder pretraining")
    logger.info("=" * 60)
    logger.info("Pretraining settings:")
    logger.info(
        f"   Current-period prediction: "
        f"{'enabled' if config.enable_current_period_in_pretraining else 'disabled'}"
    )
    logger.info(
        f"   Next-period prediction: "
        f"{'enabled' if config.enable_next_period_in_pretraining else 'disabled'}"
    )
    if config.pretraining_resume_checkpoint_path:
        logger.info(f"   Resume checkpoint: {config.pretraining_resume_checkpoint_path}")
    logger.info("-" * 60)

    # Pretraining uses inter-account attention, so use pretraining_max_accounts
    pretraining_dataset = MaskedAccountDataset(
        companies_data,
        AutoTokenizer.from_pretrained(config.account_model),
        mask_probability=config.mask_probability,
        max_length=PRETRAINING_ACCOUNT_NAME_MAX_LENGTH,
        max_accounts=config.pretraining_max_accounts,
        whole_account_masking=config.whole_account_masking,
        change_classes=config.change_classes,
        change_class_mask_token=config.change_class_mask_token,
    )

    # --- train/val split ---------------------------------------------- #
    if config.pretraining_resume_checkpoint_path:
        if split_indices is None:
            raise FileNotFoundError(
                f"Resume was requested but split_indices.json was not found: "
                f"{_split_indices_path(config)}\n"
                "Either start pretraining from scratch or restore split_indices.json."
            )
        # Resume: reuse the saved split (keeps dataset partitioning consistent)
        train_indices, val_indices = split_indices
        logger.info(
            f"Reusing saved split indices: "
            f"train={len(train_indices)}, val={len(val_indices)}"
        )
    else:
        train_indices, val_indices = create_split_indices(config, len(pretraining_dataset))

    train_dataset = Subset(pretraining_dataset, train_indices)
    val_dataset = Subset(pretraining_dataset, val_indices)
    logger.info(f"Pretraining data: train={len(train_dataset)}, val={len(val_dataset)}")

    def make_dataloader(dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=config.pretraining_batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            collate_fn=pretraining_dataset.collate_fn,
        )

    # --- Pretraining config (swap learning rate, batch size, and epochs only) --- #
    pretraining_config = dataclasses.replace(
        config,
        learning_rate=config.pretraining_learning_rate,
        batch_size=config.pretraining_batch_size,
        max_epochs=config.pretraining_epochs,
    )
    pretrainer = AccountPretrainer(pretraining_config)

    # --- Trainer ------------------------------------------------------- #
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.model_dir, "pretraining"),
        filename="account_encoder_pretrained-{epoch:02d}-{val_total_loss:.4f}",
        monitor="val_total_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor="val_total_loss", patience=3, mode="min")
    csv_logger = CSVLogger(
        save_dir=config.output_data_dir,
        name=config.model_name + "_pretraining_logs",
        flush_logs_every_n_steps=50,
    )

    trainer = pl.Trainer(
        max_epochs=config.pretraining_epochs,
        callbacks=[early_stopping, checkpoint_callback, CustomProgressBar()],
        logger=[csv_logger],
        accelerator=config.accelerator,
        devices=1,
        gradient_clip_val=0.1,  # Stronger gradient clipping than CLIP training
        gradient_clip_algorithm="norm",
        deterministic=True,
        precision=config.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=5,
        enable_checkpointing=True,
    )

    start_time = time.time()
    trainer.fit(
        pretrainer,
        make_dataloader(train_dataset, shuffle=True),
        make_dataloader(val_dataset, shuffle=False),
        ckpt_path=config.pretraining_resume_checkpoint_path or None,
    )
    elapsed = time.time() - start_time
    logger.info(f"Pretraining complete. Elapsed: {elapsed:.2f}s")

    account_encoder = _move_encoder_to_device(pretrainer.account_encoder, config.accelerator)

    # Also save the encoder alone, separately from the checkpoint
    encoder_save_path = os.path.join(config.model_dir, "account_encoder_final.pth")
    try:
        os.makedirs(config.model_dir, exist_ok=True)
        torch.save(account_encoder.state_dict(), encoder_save_path)
        logger.info(f"Saved pretrained encoder: {encoder_save_path}")
    except Exception as e:
        logger.warning(f"Encoder save error: {e}")

    return account_encoder, elapsed, (train_indices, val_indices)


def _move_encoder_to_device(encoder: torch.nn.Module, accelerator: str) -> torch.nn.Module:
    """Move the encoder to the device implied by the accelerator setting."""
    try:
        if accelerator == "cpu":
            logger.info("Moving pretrained account encoder to CPU")
            return encoder.cpu()
        if accelerator == "gpu":
            logger.info("Moving pretrained account encoder to GPU")
            return encoder.cuda()
        logger.info("Leaving pretrained account encoder on auto-selected device")
        return encoder
    except Exception as e:
        logger.warning(f"Device move error: {e}")
        logger.info("Fallback: moving to CPU")
        return encoder.cpu()


# =============================================================================
# Save results
# =============================================================================

def save_results(
    config: MultiAccountCLIPConfig,
    model: LightningMultiAccountFinancialCLIP,
    test_results: Dict,
    companies_data: List[Dict],
    texts: List[str],
    times: Dict[str, float],
) -> None:
    """Save evaluation results and sample embeddings."""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.output_data_dir, exist_ok=True)

    metric_keys = [
        "top1_accuracy", "top3_accuracy", "top5_accuracy",
        "avg_positive_similarity", "avg_negative_similarity", "similarity_gap",
    ]
    results = {
        "config": config.__dict__,
        "test_results": {key: float(test_results[key]) for key in metric_keys},
        **times,
        "timestamp": timestamp,
    }

    results_path = os.path.join(config.output_data_dir, f"training_results_{timestamp}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # Sample embeddings (for visual inspection of training results)
    logger.info("Saving sample embeddings...")
    sample_companies = companies_data[:NUM_SAMPLE_EMBEDDINGS]
    sample_texts = texts[:NUM_SAMPLE_EMBEDDINGS]

    company_embeddings, text_embeddings = model.extract_embeddings(
        [company["account_names"] for company in sample_companies],
        [company["change_classes"] for company in sample_companies],
        sample_texts,
    )

    embeddings_path = os.path.join(config.output_data_dir, f"sample_embeddings_{timestamp}.pkl")
    with open(embeddings_path, "wb") as f:
        pickle.dump({
            "company_embeddings": company_embeddings.numpy(),
            "text_embeddings": text_embeddings.numpy(),
            "sample_companies": sample_companies,
            "sample_texts": sample_texts,
            "config": config.__dict__,
        }, f)

    logger.info(f"Results file   : {results_path}")
    logger.info(f"Embeddings file: {embeddings_path}")
    logger.info(f"Model directory: {config.model_dir}")


# =============================================================================
# main
# =============================================================================

def main():
    args = parse_args()
    config = build_config(args)

    logger.info("Starting PyTorch Lightning Multi-Account CLIP training")
    logger.info("=" * 60)
    log_config(config)

    # --- Data loading ------------------------------------------------- #
    logger.info("Loading real data")
    companies_data, texts, text_types, docids = load_real_dataset(
        config.data_dir,
        config.nrows,
        config.response_tbl_file,
        config.account_amounts_file,
        config.text_file,
        edinet_code_column=config.edinet_code_column,
        is_edgar=config.is_edgar,
    )

    logger.info(f"Companies: {len(companies_data)}")
    logger.info(f"Texts: {len(texts)}")

    account_counts = [len(company["account_names"]) for company in companies_data]
    logger.info(
        f"Accounts - mean: {np.mean(account_counts):.1f}, "
        f"min: {min(account_counts)}, max: {max(account_counts)}"
    )

    # --- step1: Pretraining ------------------------------------------- #
    pretrained_account_encoder = None
    pretraining_time = 0.0
    split_indices = load_split_indices(config)

    if config.pretrained_model_path:
        logger.info("Loading pretrained model")
        logger.info("=" * 60)
        pretrained_account_encoder = load_pretrained_account_encoder(
            config.pretrained_model_path, config
        )
        if pretrained_account_encoder is not None:
            logger.info("Successfully loaded pretrained model")
        else:
            logger.warning("Failed to load pretrained model. Will run pretraining.")
            config.enable_pretraining = True

    # If joint optimization is on and pretraining is off, skip the separate pretraining phase
    skip_pretraining = config.use_joint_pretraining_loss and not config.enable_pretraining
    if skip_pretraining:
        logger.info("Joint Optimization enabled (no separate pretraining); skipping pretraining phase")

    if config.enable_pretraining and pretrained_account_encoder is None and not skip_pretraining:
        pretrained_account_encoder, pretraining_time, split_indices = run_pretraining(
            config, companies_data, split_indices
        )

    # --- step2: CLIP training ----------------------------------------- #
    logger.info("Creating CLIP model...")
    model = LightningMultiAccountFinancialCLIP(config, pretrained_account_encoder)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters    : {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("Creating data module...")
    if split_indices is not None:
        logger.info("Reusing pretraining train/val split for CLIP training")
    data_module = MultiAccountDataModule(
        companies_data,
        texts,
        model.text_encoder.tokenizer,
        config,
        text_types=text_types,
        docids=docids,
        use_text_type_batching=True,
        # Alternating optimization + Gradient Accumulation as effective joint optimization
        use_mixed_batching=False,
        pretraining_split_indices=split_indices,
    )

    logger.info("Setting up Trainer...")
    trainer = setup_trainer(config)

    logger.info("Starting CLIP training...")
    if config.clip_checkpoint_path:
        logger.info(f"Resuming from checkpoint: {config.clip_checkpoint_path}")
        # PyTorch 2.6+ defaults to weights_only=True, so register custom classes
        # found in the checkpoint as safe globals
        torch.serialization.add_safe_globals([MultiAccountCLIPConfig])

    start_time = time.time()
    trainer.fit(model, data_module, ckpt_path=config.clip_checkpoint_path)
    clip_training_time = time.time() - start_time

    total_training_time = pretraining_time + clip_training_time
    logger.info(f"CLIP training complete. Elapsed: {clip_training_time:.2f}s")
    logger.info(
        f"Total training time: {total_training_time:.2f}s "
        f"(pretraining: {pretraining_time:.2f}s, CLIP: {clip_training_time:.2f}s)"
    )

    # --- Evaluation --------------------------------------------------- #
    # data_module.setup() already ran inside trainer.fit (re-running does not change the split)
    logger.info("Evaluating on test data...")
    test_results = evaluate_model(model, data_module.test_dataloader())

    logger.info("Final evaluation results")
    logger.info("=" * 40)
    logger.info(f"Top-1 accuracy      : {test_results['top1_accuracy']:.4f}")
    logger.info(f"Top-3 accuracy      : {test_results['top3_accuracy']:.4f}")
    logger.info(f"Top-5 accuracy      : {test_results['top5_accuracy']:.4f}")
    logger.info(f"Positive similarity : {test_results['avg_positive_similarity']:.4f}")
    logger.info(f"Negative similarity : {test_results['avg_negative_similarity']:.4f}")
    logger.info(f"Similarity gap      : {test_results['similarity_gap']:.4f}")

    save_results(
        config, model, test_results, companies_data, texts,
        times={
            "pretraining_time": pretraining_time,
            "clip_training_time": clip_training_time,
            "total_training_time": total_training_time,
        },
    )

    logger.info("All done.")

    return model, trainer, test_results


if __name__ == "__main__":
    main()
