"""
Multi-Account CLIP 共通モジュール

会計表現学習モデル（勘定科目共起パターンの事前学習 + テキストとのCLIP学習）の
モデル・データセット・設定をまとめたパッケージ。

src_a6000/multi_account_clip_common から必要な部分のみを移行したもので、
以下のアブレーション・未使用機能は含まれない:
  - 勘定科目を連結して単一BERTに通すアブレーション（use_concatenated_accounts_text）
  - 同一企業対照学習のアブレーション（contrastive_task=same_company）
  - Set Transformer による集約 / 軽量エンコーダー

金額増減クラス（ラベル）はハードコードせず config で指定する。
デフォルトは config.DEFAULT_CHANGE_CLASSES を参照。
"""

from .config import (
    DEFAULT_CHANGE_CLASSES,
    DEFAULT_CHANGE_CLASS_MASK_TOKEN,
    MultiAccountCLIPConfig,
    build_change_class_vocab,
    change_class_vocab_from_config,
    normalize_change_classes,
    parse_args,
)
from .models import (
    CustomProgressBar,
    MaskedTokenPrediction,
    MultiAccountEncoder,
    TextEncoder,
    build_account_encoder,
)
from .graph_attention import (
    GraphAttentionLayer,
    GraphAttentionPooling,
    GraphAttentionReadout,
    build_parent_child_adjacency,
)
from .lightning_modules import AccountPretrainer, LightningMultiAccountFinancialCLIP
from .datasets import (
    MaskedAccountDataset,
    MixedTextTypeBatchSampler,
    MultiAccountDataset,
    TextTypeBatchSampler,
)
from .data_module import MultiAccountDataModule
from .utils import (
    evaluate_model,
    load_pretrained_account_encoder,
    load_real_dataset,
    setup_trainer,
)

__all__ = [
    # Config
    "MultiAccountCLIPConfig",
    "parse_args",
    "DEFAULT_CHANGE_CLASSES",
    "DEFAULT_CHANGE_CLASS_MASK_TOKEN",
    "normalize_change_classes",
    "build_change_class_vocab",
    "change_class_vocab_from_config",
    # Models
    "CustomProgressBar",
    "MaskedTokenPrediction",
    "MultiAccountEncoder",
    "TextEncoder",
    "build_account_encoder",
    # Graph Attention
    "GraphAttentionLayer",
    "GraphAttentionPooling",
    "GraphAttentionReadout",
    "build_parent_child_adjacency",
    # Lightning Modules
    "AccountPretrainer",
    "LightningMultiAccountFinancialCLIP",
    # Datasets
    "MaskedAccountDataset",
    "MultiAccountDataset",
    "MixedTextTypeBatchSampler",
    "TextTypeBatchSampler",
    # Data Module
    "MultiAccountDataModule",
    # Utils
    "load_real_dataset",
    "load_pretrained_account_encoder",
    "setup_trainer",
    "evaluate_model",
]
