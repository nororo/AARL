"""
Multi-Account CLIP utility functions

- load_real_dataset              : Load real data (with caching)
- load_pretrained_account_encoder: Load a pretrained account encoder
- setup_trainer                  : Configure the PyTorch Lightning Trainer
- evaluate_model                 : Evaluate CLIP retrieval performance
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import MultiAccountCLIPConfig
from .lightning_modules import AccountPretrainer, LightningMultiAccountFinancialCLIP
from .models import CustomProgressBar, MultiAccountEncoder, build_account_encoder

logger = logging.getLogger(__name__)

# Default parent key assigned when calc_parent_key is missing for an account
DEFAULT_PARENT_KEY = "top_account"


def _make_cache_key(
    *file_paths: Path,
    nrows: Optional[int],
    is_edgar: bool,
    edinet_code_column: str = "response_edinetCode",
) -> str:
    """Build an MD5 cache key from input file paths, mtimes, and parameters."""
    hasher = hashlib.md5()
    for fp in file_paths:
        fp = Path(fp)
        hasher.update(str(fp.resolve()).encode())
        if fp.exists():
            stat = fp.stat()
            hasher.update(str(stat.st_mtime).encode())
            hasher.update(str(stat.st_size).encode())
    hasher.update(f"nrows={nrows}".encode())
    hasher.update(f"is_edgar={is_edgar}".encode())
    hasher.update(f"edinet_code_column={edinet_code_column}".encode())
    return hasher.hexdigest()[:16]


def _scalar(value):
    """Take the first element when response_tbl.loc[] returns a Series due to duplicate index."""
    return value.iloc[0] if isinstance(value, pd.Series) else value


def load_real_dataset(
    data_dir: str,
    nrows: Optional[int] = None,
    response_tbl_file: str = "response_tbl_dataset_train.pkl",
    account_amounts_file: str = "df_amounts_change_train.csv",
    text_file: str = "text_512_train.csv",
    edinet_code_column: str = "response_edinetCode",
    is_edgar: bool = False,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[List[Dict], List[str], List[str], List[str]]:
    """Load the real dataset (includes text_type, docid, and next-period data).

    Args:
        data_dir: Path to the data directory
        nrows: Limit on rows to load (for debugging; None means all rows)
        response_tbl_file: response_tbl filename
        account_amounts_file: Account amount-change data filename
        text_file: Text data filename
        edinet_code_column: response_tbl column used as company ID
        is_edgar: Whether to rename columns for the EDGAR dataset (e.g. cik_year_next → docID_next)
        cache_dir: Cache directory for intermediate data (defaults to data_dir/cache)
        use_cache: Whether to use the cache

    Returns:
        companies_data: List of company data dicts
        texts: List of texts
        text_types: List of text types
        docids: List of document IDs
    """
    proj_dir = Path(data_dir)

    # ------------------------------------------------------------------ #
    # Cache lookup
    # ------------------------------------------------------------------ #
    cache_file: Optional[Path] = None
    if use_cache:
        cache_dir_path = Path(cache_dir) if cache_dir else proj_dir / "cache"
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_key = _make_cache_key(
            proj_dir / response_tbl_file,
            proj_dir / account_amounts_file,
            proj_dir / text_file,
            nrows=nrows,
            is_edgar=is_edgar,
            edinet_code_column=edinet_code_column,
        )
        cache_file = cache_dir_path / f"load_real_dataset_{cache_key}.pkl"
        if cache_file.exists():
            logger.info(f"Loading intermediate data from cache: {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    companies_data, texts, text_types, docids = pickle.load(f)
                logger.info(
                    f"Cache load complete: {len(companies_data):,} samples "
                    f"(unique docids: {len(set(docids)):,})"
                )
                return companies_data, texts, text_types, docids
            except Exception as e:
                logger.warning(f"Failed to load cache (will reprocess): {e}")

    # ------------------------------------------------------------------ #
    # File loading
    # ------------------------------------------------------------------ #
    response_tbl = pd.read_pickle(proj_dir / response_tbl_file)

    if is_edgar:
        response_tbl = response_tbl.rename(columns={"cik_year_next": "docID_next"}).set_index("docid")

    # Drop duplicate index entries (duplicates make loc[] return a Series)
    n_before = len(response_tbl)
    response_tbl = response_tbl[~response_tbl.index.duplicated(keep="first")]
    if n_before != len(response_tbl):
        logger.warning(
            f"Removed {n_before - len(response_tbl)} duplicate index entries from response_tbl."
        )

    if nrows is not None:
        response_tbl = response_tbl.sample(nrows)

    account_amounts_change_df = pd.read_csv(proj_dir / account_amounts_file)
    text_df = pd.read_csv(proj_dir / text_file)

    has_text_type = "text_type" in text_df.columns
    if not has_text_type:
        logger.warning("text_df has no text_type column; treating text_type as 'unknown'.")

    has_docid_next = "docID_next" in response_tbl.columns
    if has_docid_next:
        logger.info("Detected docID_next column; loading next-period data.")
    else:
        logger.info("docID_next column not found; next-period data will not be used.")

    has_edinet_code = edinet_code_column in response_tbl.columns
    if has_edinet_code:
        logger.info(f"Detected {edinet_code_column} column; using it as company ID.")
    else:
        logger.warning(
            f"{edinet_code_column} column not found; falling back to docid as company ID."
        )

    # Pre-group by docid (O(n²) linear search → O(1) lookup)
    account_groups = {docid: grp for docid, grp in account_amounts_change_df.groupby("docid")}
    text_groups = {docid: grp for docid, grp in text_df.groupby("docid")}
    available_docids = set(account_groups.keys())

    companies_data: List[Dict] = []
    texts: List[str] = []
    text_types: List[str] = []
    docids: List[str] = []

    next_period_count = 0
    next_period_missing_count = 0

    for docid in tqdm(response_tbl.index.tolist(), desc="Processing companies", unit="co"):
        if docid not in available_docids or docid not in text_groups:
            continue

        doc_df = account_groups[docid]
        account_label_col = (
            "label_en_filled"
            if is_edgar and "label_en_filled" in doc_df.columns
            else "label_jp_long_filled"
        )
        account_names = doc_df[account_label_col].tolist()
        change_classes = doc_df["amounts_change_cls"].tolist()

        # Columns for parent–child graph construction (calc_parent_key links to the key column)
        calc_parent_keys = (
            doc_df["calc_parent_key"].fillna(DEFAULT_PARENT_KEY).tolist()
            if "calc_parent_key" in doc_df.columns else None
        )
        account_keys = doc_df["key"].fillna("").tolist() if "key" in doc_df.columns else None

        # Next-period data
        account_names_next = None
        change_classes_next = None
        account_keys_next = None
        docid_next = None

        if has_docid_next:
            docid_next_value = _scalar(response_tbl.loc[docid, "docID_next"])
            if (
                pd.notna(docid_next_value)
                and docid_next_value != ""
                and docid_next_value in available_docids
            ):
                docid_next = docid_next_value
                doc_df_next = account_groups[docid_next]
                account_names_next = doc_df_next[account_label_col].tolist()
                change_classes_next = doc_df_next["amounts_change_cls"].tolist()
                if "key" in doc_df_next.columns:
                    account_keys_next = doc_df_next["key"].fillna("").tolist()
                next_period_count += 1
            else:
                next_period_missing_count += 1

        edinet_code = docid
        if has_edinet_code:
            edinet_code_value = _scalar(response_tbl.loc[docid, edinet_code_column])
            if pd.notna(edinet_code_value) and edinet_code_value != "":
                edinet_code = str(edinet_code_value)

        # If a docid has multiple texts (text_types), create one sample per row
        for _, row in text_groups[docid].iterrows():
            texts.append(row["text_list"])
            text_types.append(row["text_type"] if has_text_type else "unknown")
            docids.append(docid)
            companies_data.append({
                "account_names": account_names,
                "change_classes": change_classes,
                "docid": docid,
                "calc_parent_keys": calc_parent_keys,
                "account_keys": account_keys,
                "edinet_code": edinet_code,
                "account_names_next": account_names_next,
                "change_classes_next": change_classes_next,
                "account_keys_next": account_keys_next,
                "docid_next": docid_next,
            })

    logger.info(f"Data load complete: {len(companies_data)} companies")
    if has_text_type:
        logger.info(f"text_type distribution: {pd.Series(text_types).value_counts().to_dict()}")
    else:
        logger.info("text_type distribution: all 'unknown' (no text_type column)")
    logger.info(f"Unique docids: {len(set(docids))}")

    if has_docid_next:
        total_unique_docids = max(len(set(docids)), 1)
        logger.info("Next-period data stats:")
        logger.info(
            f"   - with next-period data: {next_period_count} "
            f"({100 * next_period_count / total_unique_docids:.1f}%)"
        )
        logger.info(
            f"   - missing next-period data: {next_period_missing_count} "
            f"({100 * next_period_missing_count / total_unique_docids:.1f}%)"
        )

    # ------------------------------------------------------------------ #
    # Cache save
    # ------------------------------------------------------------------ #
    if use_cache and cache_file is not None:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(
                    (companies_data, texts, text_types, docids), f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info(f"Saved intermediate data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache (results are still valid): {e}")

    return companies_data, texts, text_types, docids


def load_pretrained_account_encoder(
    model_path: str, config: MultiAccountCLIPConfig
) -> Optional[MultiAccountEncoder]:
    """Load a pretrained account encoder.

    Args:
        model_path: Path to the pretrained model (.ckpt / .pth / .pt)
        config: Configuration

    Returns:
        Loaded account encoder, or None on failure
    """
    try:
        logger.info(f"Loading pretrained model: {model_path}")

        if model_path.endswith(".ckpt"):
            # PyTorch Lightning checkpoint
            pretrainer = AccountPretrainer.load_from_checkpoint(
                model_path, config=config, weights_only=False
            )
            account_encoder = pretrainer.account_encoder

            # MLM head is pretraining-only; discard it
            if hasattr(account_encoder, "mlm_head"):
                delattr(account_encoder, "mlm_head")
            account_encoder.enable_mlm_pretraining = False

            logger.info("Loaded encoder from PyTorch Lightning checkpoint")

        elif model_path.endswith((".pth", ".pt")):
            # Raw state_dict
            account_encoder = build_account_encoder(config, enable_mlm_pretraining=False)
            state_dict = torch.load(model_path, map_location="cpu")

            # If this is an AccountPretrainer state_dict, extract the account_encoder part
            if "account_encoder.bert.embeddings.word_embeddings.weight" in state_dict:
                state_dict = {
                    key[len("account_encoder."):]: value
                    for key, value in state_dict.items()
                    if key.startswith("account_encoder.")
                }

            account_encoder.load_state_dict(state_dict, strict=False)
            logger.info("Loaded encoder from PyTorch state_dict")

        else:
            raise ValueError(f"Unsupported file format: {model_path}")

        # Reinitialize any diverged parameters
        for name, param in account_encoder.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.warning(f"NaN/Inf detected in parameter {name}, reinitializing")
                nn.init.xavier_uniform_(param.data, gain=0.1)

        return account_encoder.cpu()

    except Exception as e:
        logger.error(f"Failed to load pretrained model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def setup_trainer(config: MultiAccountCLIPConfig) -> pl.Trainer:
    """Configure the PyTorch Lightning Trainer for CLIP training."""
    os.makedirs(config.model_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config.model_dir,
        filename="multi_account_clip-{epoch:02d}-{val_loss:.4f}",
        save_top_k=config.save_top_k,
        mode="min",
        save_last=True,
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=config.patience, mode="min")

    csv_logger = CSVLogger(
        save_dir=config.output_data_dir,
        name=config.model_name + "_logs",
        flush_logs_every_n_steps=100,
    )

    return pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback, early_stopping, CustomProgressBar()],
        logger=[csv_logger],
        accelerator=config.accelerator,
        devices=1,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        deterministic=True,
        precision=config.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=50,
        val_check_interval=config.val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
    )


def evaluate_model(
    model: LightningMultiAccountFinancialCLIP,
    dataloader: DataLoader,
) -> Dict[str, float]:
    """Evaluate retrieval performance between company and text representations.

    Returns:
        Dict with top-k accuracy, average similarities, and the similarity matrix
    """
    model.eval()

    all_company_features: List[torch.Tensor] = []
    all_text_features: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            text_inputs = {
                key: value.to(model.device) for key, value in batch["text_inputs"].items()
            }
            company_features, text_features = model(
                batch["account_names_list"],
                batch["change_classes_list"],
                text_inputs,
                batch.get("calc_parent_keys_list"),
                batch.get("account_keys_list"),
            )

            all_company_features.append(company_features.cpu())
            all_text_features.append(text_features.cpu())

    if not all_company_features:
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "avg_positive_similarity": 0.0,
            "avg_negative_similarity": 0.0,
            "similarity_gap": 0.0,
            "similarity_matrix": np.empty((0, 0)),
        }

    company_features = torch.cat(all_company_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)

    # Diagonal entries are positives (same-sample company and text representations)
    similarity_matrix = torch.matmul(company_features, text_features.T).numpy()
    n = similarity_matrix.shape[0]

    def compute_topk_accuracy(k: int) -> float:
        correct = sum(
            1 for i in range(n) if i in np.argsort(similarity_matrix[i])[-k:]
        )
        return correct / n

    avg_positive_similarity = float(np.mean(np.diag(similarity_matrix)))
    off_diagonal = ~np.eye(n, dtype=bool)
    avg_negative_similarity = (
        float(np.mean(similarity_matrix[off_diagonal])) if n > 1 else 0.0
    )

    return {
        "top1_accuracy": compute_topk_accuracy(1),
        "top3_accuracy": compute_topk_accuracy(3),
        "top5_accuracy": compute_topk_accuracy(5),
        "avg_positive_similarity": avg_positive_similarity,
        "avg_negative_similarity": avg_negative_similarity,
        "similarity_gap": avg_positive_similarity - avg_negative_similarity,
        "similarity_matrix": similarity_matrix,
    }
