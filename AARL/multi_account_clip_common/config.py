"""
Multi-Account CLIP module

MultiAccountCLIPConfig (configuration data class) and parse_args (YAML + CLI parsing) are provided.

"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Definition of amount change classes (labels)
# =============================================================================
# Values appearing in the amounts_change_cls column of the actual data (df_amounts_change_*.csv).
# This is the only place for label definition, and the model and dataset must always reference through config.
#
# The order corresponds to the row index of change_class_embedding.
#     When loading existing checkpoints, do not change the order.
DEFAULT_CHANGE_CLASSES: Tuple[str, ...] = (
    "stable",
    "decrease_20",
    "increase_20",
    "increase_10",
    "decrease_10",
    "new",
)

# Input side mask/padding token. Always add one token at the end.
DEFAULT_CHANGE_CLASS_MASK_TOKEN: str = "[MASK]"

# Fallback index when unknown label string is given
# (first class in change_classes. Default is "stable").
CHANGE_CLASS_FALLBACK_INDEX: int = 0


def normalize_change_classes(change_classes: Optional[Any] = None) -> List[str]:
    """Normalize the setting value to a list of actual class names.

    Parameters
    ----------
    change_classes : list / tuple / comma-separated string / None
        If None or empty, use DEFAULT_CHANGE_CLASSES.

    Returns
    -------
    List[str]
        List of actual class names (order is preserved).
    """
    if change_classes is None:
        return list(DEFAULT_CHANGE_CLASSES)

    if isinstance(change_classes, str):
        items: Sequence[str] = [c.strip() for c in change_classes.split(",")]
    else:
        items = [str(c).strip() for c in change_classes]

    # Remove empty strings while preserving order and deduplicate
    normalized: List[str] = []
    for item in items:
        if item and item not in normalized:
            normalized.append(item)

    if not normalized:
        logger.warning("change_classes is empty, using default values")
        return list(DEFAULT_CHANGE_CLASSES)

    return normalized


def build_change_class_vocab(
    change_classes: Optional[Any] = None,
    mask_token: Optional[str] = None,
) -> Tuple[List[str], str]:
    """Build a vocabulary from actual classes + mask token (at the end).

    Returns
    -------
    vocab : List[str]
        [actual classes..., mask_token] in order. Corresponds one-to-one to the embedding index.
    mask_token : str
        The mask token actually used
    """
    mask_token = mask_token or DEFAULT_CHANGE_CLASS_MASK_TOKEN
    vocab = normalize_change_classes(change_classes)

    # The mask token is always placed at the end.
    # (The assumption that the number of actual classes = len(vocab) - 1 is used for loss calculation.)
    if mask_token in vocab:
        vocab.remove(mask_token)
    vocab.append(mask_token)

    return vocab, mask_token


def change_class_vocab_from_config(config: Any) -> Tuple[List[str], str]:
    """Extract the change class vocabulary from the config object.

    Use the default value of getattr to safely handle config objects without these fields.
    """
    return build_change_class_vocab(
        getattr(config, "change_classes", None),
        getattr(config, "change_class_mask_token", None),
    )


@dataclass
class MultiAccountCLIPConfig:
    """Class to manage the configuration of the Multi-Account CLIP model"""

    model_dir: str = "./results/multi_account_clip"
    output_data_dir: str = "./results/multi_account_clip"
    data_dir: str = "./data/afm"
    data_name: str = "multi_account_clip"
    nrows: Optional[int] = None
    max_length: int = 512

    response_tbl_file: str = "response_tbl_dataset_train.pkl"
    account_amounts_file: str = "df_amounts_change_train.csv"
    text_file: str = "text_512_train.csv"
    edinet_code_column: str = "response_edinetCode"
    is_edgar: bool = False

    change_classes: Sequence[str] = DEFAULT_CHANGE_CLASSES
    change_class_mask_token: str = DEFAULT_CHANGE_CLASS_MASK_TOKEN

    output_dim: int = 256
    text_model: str = "bert-base-multilingual-cased"
    account_model: str = "distilbert-base-multilingual-cased"
    temperature: float = 0.1
    freeze_text_bert: bool = False
    freeze_account_bert: bool = False
    max_accounts: int = 100              # maximum number of accounts per company for CLIP learning
    pretraining_max_accounts: int = 30   # maximum number of accounts per company for pretraining (due to memory consumption of account-level attention)

    # Account name encoding方式
    # True : BERT last_hidden_state with valid tokens for mean pooling
    # False: use the output of [CLS] token (at the beginning)
    use_account_name_mean_pooling: bool = False

    # Graph Attention (aggregation using the graph structure based on calc_parent_key)
    # False: use company_attention (MultiheadAttention + mean pooling)
    use_graph_attention: bool = False
    graph_attention_num_heads: int = 4
    graph_attention_num_layers: int = 2
    graph_attention_dropout: float = 0.1

    # LoRA
    use_text_lora: bool = False
    use_account_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None  # None のときはモデル構造から自動検出

    use_text_type_loss: bool = False
    text_type_loss_weight_mode: str = "equal"  # equal / sample_size / inverse_freq / manual
    text_type_loss_weights: Optional[Dict[str, float]] = None  # manual モード用

    use_joint_pretraining_loss: bool = False
    joint_mlm_loss_weight: float = 0.1
    joint_change_class_loss_weight: float = 0.05

    use_joint_next_period_loss: bool = False
    joint_next_mlm_loss_weight: float = 1.0
    joint_next_change_class_loss_weight: float = 0.5
    next_period_mask_probability: float = 0.5  # 翌期Joint MLM時のマスク率

    # Pretraining
    enable_pretraining: bool = True
    pretraining_epochs: int = 10
    pretraining_learning_rate: float = 1e-5
    mask_probability: float = 0.15
    whole_account_masking: bool = True   # True: mask whole account names at once / False: mask tokens one by one
    pretraining_batch_size: int = 8
    pretrained_model_path: Optional[str] = None            # skip pretraining if specified
    pretraining_resume_checkpoint_path: Optional[str] = None  # resume pretraining from the checkpoint
    clip_checkpoint_path: Optional[str] = None             # resume CLIP learning from the checkpoint

    enable_current_period_in_pretraining: bool = True   # current period MLM + current period change class prediction
    enable_next_period_in_pretraining: bool = False     # next period MLM + next period change class prediction
    # True : use inter_account_attention for AccountPretrainer exclusively
    # False: share company_attention / graph_attention_pooling of MultiAccountEncoder, and directly inherit the weights of pretraining to CLIP and inference
    use_inter_account_attention_in_pretraining: bool = True

    change_class_loss_weight: float = 1.0       # 当期変化クラス分類損失の重み
    next_period_mlm_loss_weight: float = 1.0    # 翌期損失の重み

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 32
    max_epochs: int = 50
    precision: str = "16-mixed"          # 32 / 16-mixed / bf16-mixed
    accumulate_grad_batches: int = 1
    val_check_interval: float = 1.0

    num_workers: int = 4
    patience: int = 10
    save_top_k: int = 3
    accelerator: str = "auto"            # auto / gpu / cpu

    @property
    def model_name(self) -> str:
        """実験管理用のモデル名"""
        return f"{self.data_name}_{self.output_dim}_{self.learning_rate}_{self.batch_size}"

    @property
    def change_class_vocab(self) -> List[str]:
        """実クラス + マスクトークン（末尾）の語彙。embeddingのインデックスに対応する。"""
        vocab, _ = build_change_class_vocab(self.change_classes, self.change_class_mask_token)
        return vocab


REMOVED_CONFIG_KEYS = frozenset({
    # Abbreviation: concatenate accounts and pass through a single BERT
    "use_concatenated_accounts_text",
    "concatenated_company_max_length",
    "change_class_text_map",
    # Abbreviation: contrastive learning within the same company
    "contrastive_task",
})


def _comma_separated_list(value: str) -> List[str]:
    """カンマ区切り文字列をリストに変換する（--change_classes 用）"""
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _str_to_bool(value: str) -> bool:
    return str(value).lower() not in ("false", "0", "no")


def _str_to_optional_bool(value: str) -> Optional[bool]:
    if str(value).lower() in ("null", "none", ""):
        return None
    return _str_to_bool(value)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments (YAML configuration file support)

    Priority: command line arguments > YAML configuration file > default values
    """
    parser = argparse.ArgumentParser(
        description="Multi-Account Financial CLIP training with PyTorch Lightning"
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML設定ファイルのパス（コマンドライン引数が設定ファイルの値を上書きする）",
    )

    # --- パス --------------------------------------------------------- #
    parser.add_argument("--model_dir", type=str, default="./results/multi_account_clip",
                        help="Model save directory")
    parser.add_argument("--output_data_dir", type=str, default="./results/multi_account_clip",
                        help="Output data save directory")
    parser.add_argument("--data_dir", type=str, default="./data/afm",
                        help="Input data directory")

    # --- データ ------------------------------------------------------- #
    parser.add_argument("--data_name", type=str, default="multi_account_clip",
                        help="Dataset name")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit the number of rows to read (for debugging)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length of text")
    parser.add_argument("--response_tbl_file", type=str, default="response_tbl_dataset_train.pkl",
                        help="response_tbl file name")
    parser.add_argument("--account_amounts_file", type=str, default="df_amounts_change_train.csv",
                        help="Account amounts change data file name")
    parser.add_argument("--text_file", type=str, default="text_512_train.csv",
                        help="Text data file name")
    parser.add_argument("--edinet_code_column", type=str, default="response_edinetCode",
                        help="Column name to use as company ID in response_tbl")
    parser.add_argument("--is_edgar", action="store_true", default=False,
                        help="Perform column name conversion for EDGAR dataset (cik_year_next→docID_next etc.)")

    # --- Change class ------------------------------------------------ #
    parser.add_argument(
        "--change_classes", type=_comma_separated_list, default=list(DEFAULT_CHANGE_CLASSES),
        metavar="CLS1,CLS2,...",
        help=(
            "Specify change classes as comma-separated values (YAML allows list specification)."
            f"Default: {','.join(DEFAULT_CHANGE_CLASSES)}."
            "The order corresponds to the embedding index, so do not change it when reading existing checkpoints."
        ),
    )
    parser.add_argument(
        "--change_class_mask_token", type=str, default=DEFAULT_CHANGE_CLASS_MASK_TOKEN,
        help="Input mask token for change classes (automatically added at the end of actual classes)",
    )

    # --- Model ------------------------------------------------------- #
    parser.add_argument("--output_dim", type=int, default=256, help="Output dimension")
    parser.add_argument("--text_model", type=str, default="bert-base-multilingual-cased",
                        help="Text encoder HuggingFace model name")
    parser.add_argument("--account_model", type=str, default="distilbert-base-multilingual-cased",
                        help="Account encoder HuggingFace model name")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature parameter for CLIP learning")
    parser.add_argument("--freeze_text_bert", action="store_true", help="テキストBERTを凍結")
    parser.add_argument("--freeze_account_bert", action="store_true", help="Freeze account BERT")
    parser.add_argument("--max_accounts", type=int, default=100,
                        help="Maximum number of accounts per company for CLIP learning")
    parser.add_argument("--pretraining_max_accounts", type=int, default=30,
                        help="Maximum number of accounts per company for pretraining")
    parser.add_argument("--use_account_name_mean_pooling", action="store_true",
                        help="Use mean pooling for account name encoding (default is [CLS] token)")

    # --- Graph Attention ----------------------------------------------- #
    parser.add_argument("--use_graph_attention", action="store_true",
                        help="Use Graph Attention to aggregate (based on calc_parent_key)")
    parser.add_argument("--graph_attention_num_heads", type=int, default=4,
                        help="Number of heads for Graph Attention (recommended: 4-8)")
    parser.add_argument("--graph_attention_num_layers", type=int, default=2,
                        help="Number of layers for Graph Attention (recommended: 1-3)")
    parser.add_argument("--graph_attention_dropout", type=float, default=0.1,
                        help="Dropout rate for Graph Attention")

    # --- LoRA ---------------------------------------------------------- #
    parser.add_argument("--use_text_lora", action="store_true",
                        help="Apply LoRA to text encoder")
    parser.add_argument("--use_account_lora", action="store_true",
                        help="Apply LoRA to account encoder")
    parser.add_argument("--lora_r", type=int, default=8, help="Rank of LoRA (recommended: 4-16)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Scaling factor for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA")

    # --- 学習 ---------------------------------------------------------- #
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"], help="Learning precision")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Gradient Accumulation: accumulate gradients of N batches before updating")
    parser.add_argument("--val_check_interval", type=float, default=1.0,
                        help="Validation interval (1.0=epoch, 0.5=half epoch)")

    # --- システム ------------------------------------------------------ #
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader")
    parser.add_argument("--patience", type=int, default=10, help="Early Stoppingの待機エポック数")
    parser.add_argument("--save_top_k", type=int, default=3, help="Number of top k models to save")
    parser.add_argument("--accelerator", type=str, default="auto",
                        choices=["auto", "gpu", "cpu"], help="Accelerator to use")

    # --- 事前学習 ------------------------------------------------------ #
    parser.add_argument("--enable_pretraining", action="store_true", default=True,
                        help="Enable pretraining for account encoder")
    parser.add_argument("--disable_pretraining", action="store_true",
                        help="Disable pretraining for account encoder")
    parser.add_argument("--pretraining_epochs", type=int, default=10, help="Number of epochs for pretraining")
    parser.add_argument("--pretraining_learning_rate", type=float, default=2e-5,
                        help="Learning rate for pretraining")
    parser.add_argument("--pretraining_batch_size", type=int, default=8,
                        help="Batch size for pretraining")
    parser.add_argument("--mask_probability", type=float, default=0.15, help="マスキング確率")
    parser.add_argument(
        "--whole_account_masking", type=_str_to_bool, default=True, metavar="BOOL",
        help=(
            "Mask whole account names at once in current period MLM (true/false). "
            "If false, mask tokens one by one."
        ),
    )
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path to pretrained model (skip pretraining if specified)")
    parser.add_argument("--pretraining_resume_checkpoint_path", type=str, default=None,
                        help="Path to resume pretraining from the checkpoint (last.ckpt etc.)")
    parser.add_argument("--clip_checkpoint_path", type=str, default=None,
                        help="Path to resume CLIP learning from the checkpoint (resume from the checkpoint if specified)")
    parser.add_argument("--enable_current_period_in_pretraining", action="store_true",
                        help="Enable current period prediction in pretraining")
    parser.add_argument("--enable_next_period_in_pretraining", action="store_true",
                        help="Enable next period prediction in pretraining")
    parser.add_argument(
        "--use_inter_account_attention_in_pretraining", type=_str_to_bool,
        default=True, metavar="BOOL",
        help=(
            "Use inter_account_attention exclusively for AccountPretrainer in pretraining (true/false). "
            "If false, share the aggregation layers of MultiAccountEncoder, and the weights are inherited by CLIP."
        ),
    )
    parser.add_argument("--change_class_loss_weight", type=float, default=1.0,
                        help="Weight of current period change class classification loss in pretraining")
    parser.add_argument("--next_period_mlm_loss_weight", type=float, default=1.0,
                        help="Weight of next period MLM loss in pretraining (enable_next_period_in_pretraining=true)")

    # --- Loss for CLIP learning ------------------------------------------------ #
    parser.add_argument("--use_text_type_loss", action="store_true",
                        help="Perform joint optimization by dividing loss by text_type")
    parser.add_argument("--text_type_loss_weight_mode", type=str, default="equal",
                        choices=["equal", "sample_size", "inverse_freq", "manual"],
                        help="Weighting mode for text_type loss")
    parser.add_argument("--text_type_loss_weights", type=str, default=None,
                        help='Manual weight setting (JSON format) Example: \'{"financial_statement":1.0}\'')
    parser.add_argument("--use_joint_pretraining_loss", action="store_true",
                        help="Perform joint optimization of pretraining loss (MLM + change class prediction) in CLIP learning")
    parser.add_argument("--joint_mlm_loss_weight", type=float, default=0.1,
                        help="Weight of MLM loss in joint optimization")
    parser.add_argument("--joint_change_class_loss_weight", type=float, default=0.05,
                        help="Weight of change class prediction loss in joint optimization")
    parser.add_argument("--use_joint_next_period_loss", action="store_true",
                        help="Add next period loss in joint optimization")
    parser.add_argument("--joint_next_mlm_loss_weight", type=float, default=1.0,
                        help="Weight of next period MLM loss in joint optimization")
    parser.add_argument("--joint_next_change_class_loss_weight", type=float, default=0.5,
                        help="Weight of next period change class prediction loss in joint optimization")
    parser.add_argument("--next_period_mask_probability", type=float, default=0.5,
                        help="Mask probability for next period MLM in joint optimization")

    # --- Inference ---------------------------------------------------------- #
    parser.add_argument("--inference_model_path", type=str, default=None,
                        help="Path to inference model checkpoint (.ckpt)")
    parser.add_argument("--inference_amounts_change_file", type=str,
                        default="df_amounts_change_all.csv",
                        help="Account amounts change data file name for inference")
    parser.add_argument("--inference_text_file", type=str, default=None,
                        help="Text data file name for inference (optional)")
    parser.add_argument("--inference_output_path", type=str,
                        default="./results/inference_results.pkl",
                        help="Output path for inference results (.pkl)")
    parser.add_argument("--inference_batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--inference_device", type=str, default="auto",
                        choices=["auto", "gpu", "cpu"], help="Inference device")
    parser.add_argument("--inference_normalize", type=_str_to_bool, default=True,
                        metavar="BOOL", help="Normalize inference vectors to L2 norm (true/false)")
    parser.add_argument("--inference_use_mean_pooling", type=_str_to_optional_bool,
                        default=None, metavar="BOOL_OR_NULL",
                        help="Use mean pooling for company level feature extraction (true/false/null)")
    parser.add_argument("--inference_use_pre_projection", type=_str_to_bool, default=False,
                        metavar="BOOL", help="Use pre-projection features for inference (true/false)")
    parser.add_argument("--inference_nrows", type=int, default=None,
                        help="Limit the number of rows to process for inference (for debugging)")

    args = parser.parse_args()

    # --- Merge YAML configuration file --------------------------------------- #
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        # Give priority to arguments explicitly specified on the command line (arguments that are different from the default values)
        default_args = parser.parse_args([])

        for key, value in config_dict.items():
            if key in REMOVED_CONFIG_KEYS:
                logger.debug(f" '{key}' is removed from the configuration file")
                continue
            if hasattr(args, key):
                if getattr(args, key) == getattr(default_args, key):
                    setattr(args, key, value)
            else:
                logger.warning(f" '{key}' is not a recognized argument in the configuration file")

        logger.info(f" Loaded settings from configuration file '{args.config}'")

    # change_classes is different input format in YAML and CLI, so normalize it here
    args.change_classes = normalize_change_classes(args.change_classes)

    return args
