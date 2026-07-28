# multi_account_clip_common

Shared modules for accounting representation learning (step1: pretraining on account co-occurrence patterns + step2: CLIP training with text).
Migrated from `src_a6000/multi_account_clip_common/` with only the parts that are needed.

## File layout

| File | Contents |
|------|----------|
| `config.py` | `MultiAccountCLIPConfig`, `parse_args`, change-class definitions |
| `models.py` | `MultiAccountEncoder` (company representation), `TextEncoder`, `MaskedTokenPrediction`, `build_account_encoder` |
| `graph_attention.py` | Parent–child / sibling graphs from `calc_parent_key` and GAT aggregation |
| `datasets.py` | `MaskedAccountDataset` (pretraining), `MultiAccountDataset` (CLIP), text_type batch sampler |
| `data_module.py` | `MultiAccountDataModule` (DataLoader construction for CLIP training) |
| `lightning_modules.py` | `AccountPretrainer` (step1), `LightningMultiAccountFinancialCLIP` (step2) |
| `utils.py` | Data loading, pretrained encoder loading, Trainer setup, evaluation |

## Change-class configuration

Previously both the model and datasets hard-coded
`['stable', 'decrease_20', 'increase_20', 'increase_10', 'decrease_10', 'new', '[MASK]']`.
Now `DEFAULT_CHANGE_CLASSES` in `config.py` is the single source of truth and can be overridden via config.

```yaml
# YAML
change_classes: [stable, decrease_20, increase_20, increase_10, decrease_10, new]
change_class_mask_token: "[MASK]"
```

```bash
# CLI (comma-separated)
--change_classes stable,decrease_20,increase_20,increase_10,decrease_10,new
```

- `change_class_mask_token` is used only for input masking and padding.
  It is automatically appended once at the **end** of the real classes (loss is computed over real classes only).
- Order corresponds to row indices of `change_class_embedding`.
  **Do not change the order when loading an existing checkpoint.**
- Unknown label strings fall back to the first class in `change_classes` (default: `stable`).
- Older checkpoint configs that lack these fields still work, because
  `change_class_vocab_from_config()` falls back to the defaults.

## Removed from the src_a6000 version

### Ablations (see “comparison models / ablation experiments” in `desigh_detail.md`)
- Concatenating accounts into one text and feeding a single BERT
  (`use_concatenated_accounts_text` / `concatenated_company_max_length` / `change_class_text_map`)
- Same-company contrastive learning (`contrastive_task=same_company`, `SameCompanyBatchSampler`,
  `compute_same_company_contrastive_loss`)

### Unused code
- Entire `set_transformer.py` (`use_set_transformer` is never enabled in any config)
- `LightweightTextEncoder` and `use_lightweight_*` settings
- `build_adjacency_from_parent_keys` / `build_hierarchical_adjacency`
  (only `build_parent_child_adjacency` is used)
- `create_sample_data` and `--debug` (debug runs on sample data)
- Uncalled methods:
  `_forward_with_data` / `_get_pooled_embeddings` / `_compute_next_period_contrastive_loss` /
  `_gat_pool` / `_gat_readout_batch`
- Unreferenced settings:
  `joint_next_contrastive_loss_weight` / `next_period_mask_ratio_min` /
  `next_period_mask_ratio_max` / `next_period_temperature`

Removed config keys are listed in `config.REMOVED_CONFIG_KEYS`.
Passing an existing `config_base.yaml` as-is is ignored without warnings.

## Other changes

- Promoted `model_dir` / `output_data_dir` / `data_dir` to formal fields of `MultiAccountCLIPConfig`
  (previously attached as attributes in the training scripts)
- Deduplicated loss computation in `training_step` and `validation_step` into `_shared_step`
- Merged `_apply_masking` and `_apply_masking_with_rate` into one method
- Deduplicated three DataLoader methods into `_make_dataloader`
- Removed the double `_encode_accounts` call in `forward_next_period`
- Vectorized the nested loop in `_apply_inter_account_attention`
- Removed `try/except` blocks that swallowed exceptions and returned zero vectors
  (batch skipping on CUDA errors is intentionally kept)

## Verification

`model/*.ckpt` (step1 / step2 / step2only / samecompany) load with no missing or unexpected
`state_dict` keys. For step1 and step2 checkpoints, outputs on identical inputs are
**bit-identical** between the src_a6000 version and this package.

```python
import sys
sys.path.insert(0, "AARL/src")

from multi_account_clip_common import LightningMultiAccountFinancialCLIP
import torch

ckpt = torch.load("model/account_encoder_pretrained_step2.ckpt",
                  map_location="cpu", weights_only=False)
model = LightningMultiAccountFinancialCLIP(config=ckpt["hyper_parameters"]["config"])
model.load_state_dict(ckpt["state_dict"], strict=False)
```

## Notes

Training and inference entry points
(`train_multi_account_clip_lightning_refactored.py` /
`inference_multi_account_clip_refactored.py`) were not migrated and remain under `src_a6000/`.
Existing analysis workflows such as `src/model_io.py` continue to reference `src_a6000/`.
