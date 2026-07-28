"""
Multi-Account CLIP model classes

CLIP learning and pretraining use the main encoders defined here.

- CustomProgressBar     : Customized ProgressBar for metrics display
- MaskedTokenPrediction : MLM / change class prediction heads for pretraining
- MultiAccountEncoder   : Account name × change class → company level representation
- TextEncoder           : Text → text representation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from pytorch_lightning.callbacks import TQDMProgressBar
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .config import (
    CHANGE_CLASS_FALLBACK_INDEX,
    build_change_class_vocab,
    change_class_vocab_from_config,
)
from .graph_attention import GraphAttentionPooling, build_parent_child_adjacency

logger = logging.getLogger(__name__)

# Maximum length of account name tokenization (account name is short enough to use a fixed value)
ACCOUNT_NAME_MAX_LENGTH = 64

# Embedding dimension for change class (concatenated with account name embedding and into account_fusion)
CHANGE_CLASS_EMBED_DIM = 64

# Number of heads for MultiheadAttention used for company level aggregation
COMPANY_ATTENTION_NUM_HEADS = 4


class CustomProgressBar(TQDMProgressBar):
    """ProgressBar with customized metrics display format"""

    def get_metrics(self, trainer, model):
        """Display metrics with 3 decimal places"""
        items = super().get_metrics(trainer, model)
        items = {k: f"{v:.3f}" if isinstance(v, (int, float)) else v for k, v in items.items()}
        return items


# Candidate module names for LoRA application (priority order).
# Automatically detect actual patterns from model structure.
_LORA_TARGET_PATTERNS: List[List[str]] = [
    ["attn.Wqkv"],                        # ModernBERT recommended
    ["Wqkv"],                             # ModernBERT short form
    ["q_lin", "v_lin"],                   # DistilBERT
    ["q_proj", "v_proj"],                 # Standard Transformer
    ["query", "value"],                   # BERT-family
    ["self_attn.q_proj", "self_attn.v_proj"],
]


def _detect_lora_target_modules(model: nn.Module) -> List[str]:
    """Detect LoRA application targets from model module names automatically."""
    module_names = {name for name, _ in model.named_modules()}

    for pattern in _LORA_TARGET_PATTERNS:
        if any(any(p in name for p in pattern) for name in module_names):
            logger.info(f"Detected LoRA application targets: {pattern}")
            return pattern

    logger.warning("No specific attention layers found, applying to all linear layers")
    return ["Linear"]


def _apply_lora(
    bert: nn.Module,
    encoder_label: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Optional[List[str]],
) -> nn.Module:
    """Apply LoRA to BERT. If failed, return the original model."""
    logger.info(f"Apply LoRA to {encoder_label}")

    if lora_target_modules is None:
        lora_target_modules = _detect_lora_target_modules(bert)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    try:
        bert = get_peft_model(bert, lora_config)
        bert.print_trainable_parameters()
    except Exception as e:
        logger.error(f"LoRA application error: {e}")
        logger.warning("Continue without LoRA")

    return bert


# =============================================================================
# Masked Token Prediction
# =============================================================================

class MaskedTokenPrediction(nn.Module):
    """Masked Token Prediction pretraining module for the account encoder

    Args:
        hidden_size: Hidden dimension of the input
        change_classes: Change class vocabulary (actual classes + mask token).
                        The dimension of the output head is this length.
        vocab_size: Vocabulary size of the account name prediction head
    """

    # Maximum vocabulary size for account name prediction head (memory usage reduction)
    MAX_VOCAB_SIZE = 120000

    def __init__(self, hidden_size: int, change_classes: Sequence[str], vocab_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.change_classes = list(change_classes)
        self.vocab_size = min(vocab_size, self.MAX_VOCAB_SIZE)

        # Account name mask prediction head
        self.account_name_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 4, self.vocab_size),
        )

        # Change class mask prediction head
        self.change_class_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(self.change_classes)),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights (Xavier + bias zero)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            account_name_logits: [batch_size, seq_len, vocab_size]
            change_class_logits: [batch_size, seq_len, num_change_classes]
        """
        account_name_logits = self.account_name_prediction_head(hidden_states)
        change_class_logits = self.change_class_prediction_head(hidden_states)

        return account_name_logits, change_class_logits


########################################################
# Multi-Account Encoder (Multiple accounts encoder)
########################################################

class MultiAccountEncoder(nn.Module):
    """Encoder to aggregate multiple accounts and create a company level representation

    Processing flow:
        Account name --BERT--> name embedding ┐
                                            ├-- account_fusion --> account representation
        Change class --Embedding--> change embedding ┘
                                            ↓
                    Company level aggregation (Graph Attention or company_attention)
                                            ↓
                                       projection
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        output_dim: int = 256,
        freeze_bert: bool = False,
        max_accounts: int = 100,
        enable_mlm_pretraining: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        use_graph_attention: bool = False,
        graph_attention_num_heads: int = 4,
        graph_attention_num_layers: int = 2,
        graph_attention_dropout: float = 0.1,
        use_account_name_mean_pooling: bool = False,
        change_classes: Optional[Sequence[str]] = None,
        change_class_mask_token: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.bert = AutoModel.from_pretrained(model_name)

        if use_lora and not freeze_bert:
            self.bert = _apply_lora(
                self.bert, "Account Encoder",
                lora_r, lora_alpha, lora_dropout, lora_target_modules,
            )
        elif freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("   BERT is frozen")

        self.hidden_size = self.bert.config.hidden_size

        self.max_accounts = max_accounts
        self.enable_mlm_pretraining = enable_mlm_pretraining
        self._freeze_bert = freeze_bert
        self.use_graph_attention = use_graph_attention
        self.use_account_name_mean_pooling = use_account_name_mean_pooling
        if use_account_name_mean_pooling:
            logger.info("Account name encoder: use mean pooling mode (instead of CLS token)")

        # --- Change class embedding ----------------------------------- #
        # change_classes is the vocabulary of "actual classes + mask token (at the end)".
        # The mask token is used for input masking and padding, but not for prediction labels.
        self.change_classes, self.change_class_mask_token = build_change_class_vocab(
            change_classes, change_class_mask_token
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.change_classes)}
        self.change_class_mask_idx = self.class_to_idx[self.change_class_mask_token]
        self.num_real_change_classes = len(self.change_classes) - 1
        # Fallback for unknown label strings
        self.default_change_class_idx = CHANGE_CLASS_FALLBACK_INDEX
        logger.info(
            f"Change classes: {self.change_classes[:-1]} "
            f"(+ mask token '{self.change_class_mask_token}')"
        )

        self.change_class_embedding = nn.Embedding(
            len(self.change_classes), CHANGE_CLASS_EMBED_DIM
        )

        # --- Account name × change class fusion layer ----------------------------- #
        self.account_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + CHANGE_CLASS_EMBED_DIM, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # --- Company level aggregation ------------------------------------------- #
        self.graph_attention_pooling = None
        self.company_attention = None

        if use_graph_attention:
            logger.info("Using Graph Attention pooling")
            logger.info(f"   GAT heads: {graph_attention_num_heads}")
            logger.info(f"   GAT layers: {graph_attention_num_layers}")
            logger.info(f"   Dropout: {graph_attention_dropout}")

            self.graph_attention_pooling = GraphAttentionPooling(
                dim=self.hidden_size,
                num_heads=graph_attention_num_heads,
                num_layers=graph_attention_num_layers,
                dropout=graph_attention_dropout,
            )
        else:
            logger.info("Using standard attention pooling")
            self.company_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=COMPANY_ATTENTION_NUM_HEADS,
                dropout=0.1,
                batch_first=True,
            )

        # --- Final projection layer --------------------------------------------- #
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, output_dim),
            nn.LayerNorm(output_dim),
        )

        # --- Pretraining head --------------------------------------------------- #
        if enable_mlm_pretraining:
            self.mlm_head = MaskedTokenPrediction(
                hidden_size=self.hidden_size,
                change_classes=self.change_classes,
                vocab_size=self.tokenizer.vocab_size,
            )

    @property
    def output_dim(self) -> int:
        """Output dimension after projection"""
        return self.projection[-1].normalized_shape[0]

    def change_class_indices(
        self,
        change_classes: Sequence[Any],
        num_accounts: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert a list of change class names into an index tensor for embedding.

        Unknown labels fall back to default_change_class_idx.
        Pads or truncates when the length does not match num_accounts.

        Args:
            change_classes: List of change class names
            num_accounts: Number of corresponding accounts
            device: Device for the output tensor

        Returns:
            indices: [num_accounts] long tensor
        """
        indices = [
            self.class_to_idx.get(str(cls), self.default_change_class_idx)
            for cls in change_classes
        ]

        if len(indices) < num_accounts:
            indices.extend([self.default_change_class_idx] * (num_accounts - len(indices)))
        else:
            indices = indices[:num_accounts]

        return torch.tensor(indices, dtype=torch.long, device=device)

    def _tokenize_account_names(
        self,
        account_names: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize account name list and move tensors to the given device."""
        tokenized = self.tokenizer(
            account_names,
            padding=True,
            truncation=True,
            max_length=ACCOUNT_NAME_MAX_LENGTH,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in tokenized.items()}

    def _encode_account_names(self, tokenized: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pass tokenized account names through BERT to get account name embeddings.

        Returns:
            [num_accounts, hidden_size]
        """
        with torch.no_grad() if self._freeze_bert else torch.enable_grad():
            bert_outputs = self.bert(**tokenized)
            if self.use_account_name_mean_pooling:
                # Mean pooling over valid tokens only (attention_mask=1)
                last_hidden = bert_outputs.last_hidden_state          # [N, seq_len, hidden]
                mask = tokenized["attention_mask"].unsqueeze(-1).float()
                return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return bert_outputs.last_hidden_state[:, 0, :]            # [CLS] token

    def _aggregate_company(
        self,
        fused_features: torch.Tensor,
        account_names: List[str],
        calc_parent_keys: Optional[List[str]],
        account_keys: Optional[List[str]],
        device: torch.device,
    ) -> torch.Tensor:
        """Aggregate a set of account representations into a single company-level vector.

        Args:
            fused_features: [num_accounts, hidden_size]

        Returns:
            [hidden_size]
        """
        if fused_features.size(0) <= 1:
            # No aggregation needed when there is only one account
            return fused_features.squeeze(0)

        if self.use_graph_attention and self.graph_attention_pooling is not None:
            adj_mask = self._build_adjacency(
                fused_features.size(0), account_names, calc_parent_keys, account_keys, device
            )
            return self.graph_attention_pooling(fused_features, adj_mask)

        # Aggregate with standard attention, then mean pool
        aggregated_features, _ = self.company_attention(
            fused_features.unsqueeze(0),
            fused_features.unsqueeze(0),
            fused_features.unsqueeze(0),
        )
        return aggregated_features.squeeze(0).mean(dim=0)

    def _build_adjacency(
        self,
        n_nodes: int,
        account_names: List[str],
        calc_parent_keys: Optional[List[str]],
        account_keys: Optional[List[str]],
        device: torch.device,
    ) -> torch.Tensor:
        """Build the adjacency matrix for Graph Attention.

        Falls back to a fully connected graph when calc_parent_key is absent.
        """
        debug = getattr(self, "_gat_debug_count", 0) < 3
        if debug:
            self._gat_debug_count = getattr(self, "_gat_debug_count", 0) + 1
            logger.info(
                f"[GAT] sample #{self._gat_debug_count}: n_nodes={n_nodes}, "
                f"calc_parent_keys={'present' if calc_parent_keys is not None else 'absent (fully-connected fallback)'}"
            )

        if calc_parent_keys is None:
            adj_mask = torch.ones(n_nodes, n_nodes, dtype=torch.bool, device=device)
            if debug:
                logger.info(f"[GAT] using fully-connected graph: n_nodes={n_nodes}, n_edges={n_nodes * n_nodes}")
            return adj_mask

        # Match calc_parent_key against key column values to form parent-child edges.
        # Fall back to account names when account_keys is absent.
        parent_keys = list(calc_parent_keys[:n_nodes])
        parent_keys += ["top_account"] * (n_nodes - len(parent_keys))

        if account_keys is not None:
            keys_for_adj = list(account_keys[:n_nodes])
            keys_for_adj += [""] * (n_nodes - len(keys_for_adj))
        else:
            keys_for_adj = account_names

        adj_mask = build_parent_child_adjacency(
            keys_for_adj, parent_keys, device=device, debug=debug
        )
        if debug:
            num_edges = int(adj_mask.sum().item())
            logger.info(
                f"[GAT] adj_mask shape={adj_mask.shape}, "
                f"valid_edges={num_edges} / max={n_nodes * n_nodes}, "
                f"density={num_edges / (n_nodes * n_nodes):.3f}"
            )
        return adj_mask

    def forward(
        self,
        account_names_list: List[List[str]],
        change_classes_list: List[List[str]],
        calc_parent_keys_list: Optional[List[Optional[List[str]]]] = None,
        account_keys_list: Optional[List[Optional[List[str]]]] = None,
        return_pre_projection: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            account_names_list: Per-company account name lists in the batch
            change_classes_list: Per-company change class lists in the batch
            calc_parent_keys_list: Per-company calc_parent_key lists (when using Graph Attention)
            account_keys_list: Per-company key column values (matched with calc_parent_key for parent-child edges)
            return_pre_projection: If True, skip projection and return hidden_size-dimensional features

        Returns:
            company_features: [batch_size, output_dim] or [batch_size, hidden_size]
        """
        device = next(self.parameters()).device
        out_dim = self.hidden_size if return_pre_projection else self.output_dim
        batch_size = len(account_names_list)

        # On length mismatch, truncate to the shorter side
        if batch_size != len(change_classes_list):
            logger.error(
                f"Length mismatch: account_names={batch_size}, "
                f"change_classes={len(change_classes_list)}"
            )
            batch_size = min(batch_size, len(change_classes_list))
            account_names_list = account_names_list[:batch_size]
            change_classes_list = change_classes_list[:batch_size]

        if batch_size == 0:
            return torch.zeros(0, out_dim, device=device)

        company_features: List[torch.Tensor] = []

        for i in range(batch_size):
            account_names = self._clean_account_names(account_names_list[i])

            if not account_names:
                # Companies with no valid accounts get a zero vector
                company_features.append(torch.zeros(self.hidden_size, device=device))
                continue

            tokenized = self._tokenize_account_names(account_names, device)
            account_text_features = self._encode_account_names(tokenized)

            # Embed amount change classes
            change_indices = self.change_class_indices(
                change_classes_list[i], len(account_names), device
            )
            change_features = self.change_class_embedding(change_indices)

            # Fuse account name × change class
            fused_features = self.account_fusion(
                torch.cat([account_text_features, change_features], dim=-1)
            )

            company_features.append(
                self._aggregate_company(
                    fused_features,
                    account_names,
                    calc_parent_keys_list[i] if calc_parent_keys_list is not None
                    and i < len(calc_parent_keys_list) else None,
                    account_keys_list[i] if account_keys_list is not None
                    and i < len(account_keys_list) else None,
                    device,
                )
            )

        stacked = torch.stack(company_features)  # [batch_size, hidden_size]

        if return_pre_projection:
            return stacked
        return self.projection(stacked)

    @staticmethod
    def _clean_account_names(account_names: Any) -> List[str]:
        """Normalize account name list to strings and drop empty entries."""
        if not isinstance(account_names, (list, tuple)):
            logger.error(f"account_names should be a list, got {type(account_names)}")
            account_names = [account_names] if account_names else []

        cleaned = [str(name).strip() for name in account_names if name is not None]
        return [name for name in cleaned if name]


def build_account_encoder(
    config: Any,
    enable_mlm_pretraining: bool = False,
    freeze_bert: Optional[bool] = None,
) -> MultiAccountEncoder:
    """Build MultiAccountEncoder from config (shared by pretraining, CLIP training, and inference).

    Args:
        config: MultiAccountCLIPConfig (uses getattr so older checkpoint configs still work)
        enable_mlm_pretraining: Whether to create the MLM head
        freeze_bert: If None, follow config.freeze_account_bert
    """
    change_classes, change_class_mask_token = change_class_vocab_from_config(config)

    return MultiAccountEncoder(
        model_name=config.account_model,
        output_dim=config.output_dim,
        freeze_bert=config.freeze_account_bert if freeze_bert is None else freeze_bert,
        max_accounts=config.max_accounts,
        enable_mlm_pretraining=enable_mlm_pretraining,
        use_lora=config.use_account_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        use_graph_attention=config.use_graph_attention,
        graph_attention_num_heads=config.graph_attention_num_heads,
        graph_attention_num_layers=config.graph_attention_num_layers,
        graph_attention_dropout=config.graph_attention_dropout,
        use_account_name_mean_pooling=getattr(config, "use_account_name_mean_pooling", False),
        change_classes=change_classes,
        change_class_mask_token=change_class_mask_token,
    )


# =============================================================================
# Text Encoder
# =============================================================================

class TextEncoder(nn.Module):
    """Encoder for securities report (yukashoken hokokusho) text"""

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        output_dim: int = 256,
        freeze_bert: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if use_lora and not freeze_bert:
            self.bert = _apply_lora(
                self.bert, "Text Encoder",
                lora_r, lora_alpha, lora_dropout, lora_target_modules,
            )
        elif freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.hidden_size = self.bert.config.hidden_size
        # Max sequence length the model can handle (rows of position embeddings).
        # Kept to avoid crashes when inputs exceed max_length.
        self.max_position_embeddings = int(
            getattr(self.bert.config, "max_position_embeddings", 512)
        )

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, output_dim),
            nn.LayerNorm(output_dim),
        )

        self._freeze_bert = freeze_bert

    def forward(
        self,
        text_inputs: Dict[str, torch.Tensor],
        return_pre_projection: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            text_inputs: Tokenized text
            return_pre_projection: If True, skip projection and return hidden_size-dimensional features

        Returns:
            text_features: [batch_size, output_dim] or [batch_size, hidden_size]
        """
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        # Clamp token IDs to a valid range
        vocab_size = self.tokenizer.vocab_size
        if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
            logger.warning(
                f"Invalid token IDs detected. "
                f"Range: [{input_ids.min()}, {input_ids.max()}], vocab_size: {vocab_size}"
            )
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            text_inputs = {**text_inputs, "input_ids": input_ids}

        # Truncate to the model's max_position_embeddings
        seq_length = input_ids.size(1)
        if seq_length > self.max_position_embeddings:
            logger.warning(
                f"Sequence length {seq_length} exceeds "
                f"max_position_embeddings {self.max_position_embeddings}, truncating"
            )
            seq_length = self.max_position_embeddings
            input_ids = input_ids[:, :seq_length]
            attention_mask = attention_mask[:, :seq_length]
            text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Create position_ids explicitly (DistilBERT does not accept position_ids)
        if "distilbert" not in self.model_name.lower():
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)
            text_inputs = {**text_inputs, "position_ids": position_ids}

        with torch.no_grad() if self._freeze_bert else torch.enable_grad():
            bert_outputs = self.bert(**text_inputs)
            text_features = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        if return_pre_projection:
            return text_features
        return self.projection(text_features)
