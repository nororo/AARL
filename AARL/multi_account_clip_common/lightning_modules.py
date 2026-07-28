"""
PyTorch Lightning Modules for Multi-Account Financial CLIP

- AccountPretrainer                  : Account encoder pretraining (step1)
- LightningMultiAccountFinancialCLIP : CLIP learning with text (step2)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from .config import MultiAccountCLIPConfig
from .graph_attention import build_parent_child_adjacency
from .models import (
    MaskedTokenPrediction,
    MultiAccountEncoder,
    TextEncoder,
    build_account_encoder,
)

logger = logging.getLogger(__name__)

# Range of logit clipping for numerical stability of loss calculation
LOGIT_CLAMP = 10.0

# Number of attention heads for inter-account attention / next period cross-attention
PRETRAIN_ATTENTION_NUM_HEADS = 8

# Maximum length of account name tokenization during joint optimization
JOINT_ACCOUNT_NAME_MAX_LENGTH = 128


class AccountPretrainer(pl.LightningModule):
    """Lightning Module for account encoder pretraining (multiple accounts supported)

    Tasks to learn:
      - Current period account name MLM   : Predict masked account names using other accounts as context
      - Current period change class prediction  : Predict masked change classes using other account change classes
      - Next period change class prediction  : Predict next period change classes from all current period account information
    """

    def __init__(self, config: MultiAccountCLIPConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # 事前学習では勘定科目BERTを必ず学習させるため freeze_bert=False で固定する
        self.account_encoder = build_account_encoder(
            config, enable_mlm_pretraining=True, freeze_bert=False
        )
        hidden_size = self.account_encoder.hidden_size

        # inter_account_attention を使うか（True で従来動作）。
        # False にすると MultiAccountEncoder 内の company_attention / graph_attention_pooling を
        # コンテキスト化にも流用し、CLIP推論フェーズと重みを共有する。
        self.use_inter_account_attention = config.use_inter_account_attention_in_pretraining

        if self.use_inter_account_attention:
            # AccountPretrainer 専用の勘定科目間attention層
            self.inter_account_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=PRETRAIN_ATTENTION_NUM_HEADS,
                dropout=0.1,
                batch_first=True,
            )
            self.inter_account_norm = nn.LayerNorm(hidden_size)
        else:
            logger.info(
                "📌 use_inter_account_attention_in_pretraining=False: "
                "事前学習のコンテキスト化に MultiAccountEncoder の "
                "company_attention / graph_attention_pooling を使用します"
            )

        # 翌期予測用のcross-attention層（当期→翌期）
        self.next_period_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=PRETRAIN_ATTENTION_NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )
        self.next_period_norm = nn.LayerNorm(hidden_size)

        # 翌期変化クラス予測ヘッド（当期情報を入力とする）
        self.next_change_class_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(self.account_encoder.change_classes)),
        )

        self._log_task_architecture_matrix()

    # ------------------------------------------------------------------ #
    # ログ
    # ------------------------------------------------------------------ #

    def _log_task_architecture_matrix(self):
        """事前学習で選択されているタスク × アーキテクチャの構成をログ出力する。"""
        arch = "Graph Attention" if self.config.use_graph_attention else "Company Attention（デフォルト）"
        use_iaa = self.use_inter_account_attention
        iaa_label = "inter_account_attention（専用）" if use_iaa else f"{arch}（共有）"


        if self.config.enable_current_period_in_pretraining:
            if not use_iaa and self.config.use_graph_attention:
                ctx = "GAT encode（親子グラフ使用）"
            else:
                ctx = iaa_label
        else:
            logger.info("  T1 当期MLM           : 無効")
            logger.info("  T2 当期Change予測    : 無効")

        if self.config.enable_next_period_in_pretraining:
            if self.config.use_graph_attention:
                ctx = "GAT encode（当期）→ masked_mean → broadcast"
            else:
                ctx = iaa_label + " → masked_mean → broadcast"
        else:
            logger.info("  T3 翌期予測          : 無効")

        logger.info("-" * 80)
        else:
        logger.info("=" * 80)

    # ------------------------------------------------------------------ #
    # エンコード
    # ------------------------------------------------------------------ #

    def _encode_accounts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        change_class_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode multiple accounts

        Same path as MultiAccountEncoder.forward() during CLIP inference
        (bert → change_class_embedding → account_fusion) to allow gradient flow to account_fusion,
        and weights to be inherited by CLIP and inference.

        Args:
            input_ids: [batch_size, num_accounts, seq_len]
            attention_mask: [batch_size, num_accounts, seq_len]
            change_class_inputs: [batch_size, num_accounts]

        Returns:
            account_embeddings: [batch_size, num_accounts, hidden_size]
            enhanced_hidden_states: [batch_size * num_accounts, seq_len, hidden_size]
        """
        encoder = self.account_encoder
        batch_size, num_accounts, seq_len = input_ids.shape

        # バッチ次元と勘定科目次元をフラット化する
        flat_input_ids = input_ids.view(batch_size * num_accounts, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_accounts, seq_len)
        flat_change_class = change_class_inputs.view(batch_size * num_accounts)
        flat_change_class = torch.clamp(flat_change_class, 0, len(encoder.change_classes) - 1)

        bert_outputs = encoder.bert(
            input_ids=flat_input_ids, attention_mask=flat_attention_mask
        )
        hidden_states = bert_outputs.last_hidden_state  # [B*N, seq_len, hidden_size]

        if encoder.use_account_name_mean_pooling:
            # Same mean pooling as CLIP inference path (MultiAccountEncoder.forward)
            mask = flat_attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states[:, 0, :]  # [CLS] token

        # Merge change class information via account_fusion (same path as CLIP inference)
        change_embeddings = encoder.change_class_embedding(flat_change_class)
        fused = encoder.account_fusion(torch.cat([pooled, change_embeddings], dim=-1))

        # Broadcast fusion difference to entire token sequence for MLM head
        fusion_delta = (fused - pooled).unsqueeze(1).expand(-1, seq_len, -1)
        enhanced_hidden_states = hidden_states + fusion_delta

        account_embeddings = fused.view(batch_size, num_accounts, encoder.hidden_size)

        return account_embeddings, enhanced_hidden_states

    def _apply_inter_account_attention(
        self,
        account_embeddings: torch.Tensor,
        account_mask: torch.Tensor,
        change_class_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply self-attention between accounts

        In current period change class prediction, masked accounts are prohibited from self-reference,
        and only information from other accounts is allowed.

        Args:
            account_embeddings: [batch_size, num_accounts, hidden_size]
            account_mask: [batch_size, num_accounts] 有効な勘定科目のマスク
            change_class_mask: [batch_size, num_accounts] change classがマスクされた勘定科目

        Returns:
            contextualized_embeddings: [batch_size, num_accounts, hidden_size]
        """
        num_accounts = account_embeddings.size(1)

        # attn_mask: True の位置は注意されない（無効な勘定科目を全行でマスク）
        attn_mask = ~account_mask.unsqueeze(1).expand(-1, num_accounts, -1).clone()

        # change class をマスクした勘定科目は自己参照を禁止する（対角要素をマスク）
        diag = torch.arange(num_accounts, device=account_embeddings.device)
        attn_mask[:, diag, diag] |= change_class_mask

        # MultiheadAttention の attn_mask は [batch*num_heads, L, S] を要求する
        attn_mask = attn_mask.repeat_interleave(PRETRAIN_ATTENTION_NUM_HEADS, dim=0)

        contextualized, _ = self.inter_account_attention(
            account_embeddings, account_embeddings, account_embeddings,
            key_padding_mask=~account_mask,
            attn_mask=attn_mask,
        )
        return self.inter_account_norm(account_embeddings + contextualized)

    def _apply_encoder_attention(
        self,
        account_embeddings: torch.Tensor,
        account_mask: torch.Tensor,
        calc_parent_keys_batch: Optional[List] = None,
        account_keys_batch: Optional[List] = None,
    ) -> torch.Tensor:
        """Use the aggregation module in MultiAccountEncoder as a contextualization layer.

        Called when use_inter_account_attention_in_pretraining=False.
        Use the same weights as the CLIP inference phase for pretraining,
        so that the effect of pretraining is directly reflected in the aggregation during inference.

        Returns:
            contextualized: [batch_size, num_accounts, hidden_size]
        """
        encoder = self.account_encoder

        if encoder.use_graph_attention and encoder.graph_attention_pooling is not None:
            return self._gat_encode_batch(
                account_embeddings, account_mask, calc_parent_keys_batch, account_keys_batch
            )

        # company_attention を contextualization として使用（residual接続）
        attended, _ = encoder.company_attention(
            account_embeddings, account_embeddings, account_embeddings,
            key_padding_mask=~account_mask,
        )
        return account_embeddings + attended

    def _gat_encode_batch(
        self,
        embeddings: torch.Tensor,
        account_mask: torch.Tensor,
        calc_parent_keys_batch: Optional[List] = None,
        account_keys_batch: Optional[List] = None,
    ) -> torch.Tensor:
        """GAT encode + global context application by batch

        Readout (company level aggregation) is not performed, and per-account contextualized representations are returned.

        Args:
            embeddings: [batch_size, num_accounts, hidden_size]
            account_mask: [batch_size, num_accounts] bool
            calc_parent_keys_batch: List[List[str] or None]（バッチ分）
            account_keys_batch: List[List[str] or None]（バッチ分、keyカラム値）

        Returns:
            [batch_size, num_accounts, hidden_size]
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        debug = getattr(self, "_pretrain_gat_debug_count", 0) < 3
        if debug:
            self._pretrain_gat_debug_count = getattr(self, "_pretrain_gat_debug_count", 0) + 1
            has_graph = calc_parent_keys_batch is not None and account_keys_batch is not None
            logger.info(
                f"[Pretrain GAT encode] 呼び出し #{self._pretrain_gat_debug_count}: "
                f"batch_size={batch_size}, "
                f"グラフ={'親子グラフ' if has_graph else '全結合フォールバック'}"
            )

        # ── Step 1: GAT encode（親子グラフ）─────────────────────────────── #
        gat_encoded = torch.zeros_like(embeddings)

        for i in range(batch_size):
            mask_i = account_mask[i]
            valid_feats = embeddings[i][mask_i]  # [n_valid, D]
            n_valid = valid_feats.size(0)

            if n_valid == 0:
                continue

            adj_mask_i = self._build_sample_adjacency(
                i, n_valid, calc_parent_keys_batch, account_keys_batch,
                device, debug=(debug and i == 0),
            )

            gat_out = self.account_encoder.graph_attention_pooling.encode(
                valid_feats, adj_mask_i
            )  # [1, n_valid, D]
            gat_encoded[i][mask_i] = gat_out.squeeze(0).to(dtype=gat_encoded.dtype)

        # ── Step 2: グローバルコンテキスト付与（クロスグループ依存）──────── #
        if self.use_inter_account_attention:
            no_change_mask = torch.zeros_like(account_mask, dtype=torch.bool)
            return self._apply_inter_account_attention(gat_encoded, account_mask, no_change_mask)

        # GAT 自体がメッセージパッシングを担っているためスキップする
        return gat_encoded

    @staticmethod
    def _build_sample_adjacency(
        sample_idx: int,
        n_valid: int,
        calc_parent_keys_batch: Optional[List],
        account_keys_batch: Optional[List],
        device: torch.device,
        debug: bool = False,
    ) -> torch.Tensor:
        """1サンプル分の隣接行列を作る（グラフ情報が無ければ全結合にフォールバック）。"""
        parent_keys = (
            calc_parent_keys_batch[sample_idx]
            if calc_parent_keys_batch is not None and sample_idx < len(calc_parent_keys_batch)
            else None
        )
        account_keys = (
            account_keys_batch[sample_idx]
            if account_keys_batch is not None and sample_idx < len(account_keys_batch)
            else None
        )

        if parent_keys is None or account_keys is None:
            if debug:
                logger.info(f"[Pretrain GAT encode] 全結合グラフ使用 (ノード数={n_valid})")
            return torch.ones(n_valid, n_valid, dtype=torch.bool, device=device)

        parent_keys = list(parent_keys[:n_valid])
        parent_keys += ["top_account"] * (n_valid - len(parent_keys))
        account_keys = list(account_keys[:n_valid])
        account_keys += [""] * (n_valid - len(account_keys))

        return build_parent_child_adjacency(
            account_keys, parent_keys, device=device, debug=debug
        )

    def _predict_next_period_change_class(
        self,
        current_account_embeddings: torch.Tensor,
        current_account_mask: torch.Tensor,
        next_account_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """翌期の変化クラスを予測する

        当期の全勘定科目情報を cross-attention で参照し、翌期の各勘定科目の変化クラスを予測する。

        Args:
            current_account_embeddings: [batch_size, N_cur, hidden_size]
            current_account_mask: [batch_size, N_cur]
            next_account_embeddings: [batch_size, N_next, hidden_size]（クエリ）

        Returns:
            next_change_logits: [batch_size, N_next, num_classes]
        """
        cross_attended, _ = self.next_period_cross_attention(
            next_account_embeddings,     # query: 翌期
            current_account_embeddings,  # key:   当期
            current_account_embeddings,  # value: 当期
            key_padding_mask=~current_account_mask,
        )
        cross_attended = self.next_period_norm(next_account_embeddings + cross_attended)

        return self.next_change_class_head(cross_attended)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """当期の勘定科目名MLM + 変化クラス予測

        Returns:
            account_name_logits: [batch_size * num_accounts, seq_len, vocab_size]
            change_class_logits: [batch_size, num_accounts, num_change_classes]
            account_embeddings : [batch_size, num_accounts, hidden_size]
        """
        input_ids = batch["input_ids"]
        account_mask = batch["account_mask"]
        batch_size, num_accounts, seq_len = input_ids.shape
        hidden_size = self.account_encoder.hidden_size

        account_embeddings, enhanced_hidden_states = self._encode_accounts(
            input_ids, batch["attention_mask"], batch["change_class_inputs"]
        )

        # 変化クラスがマスクされた勘定科目（ラベルが -100 でないもの）
        change_class_mask = batch["change_class_labels"] != -100

        # 勘定科目間のコンテキスト化
        if self.use_inter_account_attention:
            contextualized = self._apply_inter_account_attention(
                account_embeddings, account_mask, change_class_mask
            )
        else:
            contextualized = self._apply_encoder_attention(
                account_embeddings, account_mask,
                calc_parent_keys_batch=batch.get("calc_parent_keys"),
                account_keys_batch=batch.get("account_keys_str"),
            )

        # コンテキストをトークン列全体に broadcast して MLM を強化する
        context_expanded = (
            contextualized
            .view(batch_size * num_accounts, hidden_size)
            .unsqueeze(1)
            .expand(-1, seq_len, -1)
        )
        account_name_logits, _ = self.account_encoder.mlm_head(
            enhanced_hidden_states + context_expanded
        )

        # 変化クラスの予測は per-account のコンテキスト表現から行う
        change_class_logits = self.account_encoder.mlm_head.change_class_prediction_head(
            contextualized
        )

        return {
            "account_name_logits": account_name_logits,
            "change_class_logits": change_class_logits,
            "account_embeddings": contextualized,
        }

    def forward_next_period(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """翌期予測用の forward（pooled broadcast 方式）

        当期の全勘定科目を GAT / company_attention で contextualize し、masked_mean で
        1ベクトルに集約して翌期の各トークンに broadcast する。

        学習される相関:
            P(翌期の勘定科目 | 当期の全勘定科目パターン（名前・増減クラス・親子構造）)

        重みの引き継ぎ:
            bert / account_fusion / GAT・company_attention は CLIP・推論に引き継がれる
            （GAT・company_attention は use_inter_account_attention=False のとき）。

        Returns:
            next_mlm_logits           : [B*N_next, seq_len, vocab_size]
            current_account_embeddings: [B, N_cur, hidden_size]
            next_account_embeddings   : [B, N_next, hidden_size]
        """
        batch_size, num_next_accounts, seq_len = batch["next_input_ids"].shape
        account_mask = batch["account_mask"]

        # ── Step 1: 当期をエンコード → コンテキスト化 ────────────────────── #
        current_embeddings, _ = self._encode_accounts(
            batch["input_ids"], batch["attention_mask"], batch["change_class_inputs"]
        )

        if self.use_inter_account_attention:
            no_mask = torch.zeros_like(account_mask, dtype=torch.bool)
            current_ctx = self._apply_inter_account_attention(
                current_embeddings, account_mask, no_mask
            )
        else:
            current_ctx = self._apply_encoder_attention(
                current_embeddings, account_mask,
                calc_parent_keys_batch=batch.get("calc_parent_keys"),
                account_keys_batch=batch.get("account_keys_str"),
            )

        # ── Step 2: masked_mean で企業1ベクトルに集約 ───────────────────── #
        mask_f = account_mask.unsqueeze(-1).float()
        current_pooled = (current_ctx * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        # ── Step 3: 翌期をエンコード（変化クラス入力は全て[MASK]）──────── #
        next_change_class_inputs = torch.full(
            (batch_size, num_next_accounts),
            self.account_encoder.change_class_mask_idx,
            dtype=torch.long,
            device=batch["next_input_ids"].device,
        )
        next_account_emb, next_hidden = self._encode_accounts(
            batch["next_input_ids"], batch["next_attention_mask"], next_change_class_inputs
        )

        # ── Step 4: 当期 pooled を翌期の全トークンに broadcast ──────────── #
        ctx_expanded = (
            current_pooled
            .unsqueeze(1)                                     # [B, 1, D]
            .expand(-1, num_next_accounts, -1)                # [B, N_next, D]
            .reshape(batch_size * num_next_accounts, 1, -1)   # [B*N_next, 1, D]
            .expand(-1, seq_len, -1)                          # [B*N_next, seq_len, D]
        )

        # ── Step 5: MLM ヘッドで翌期勘定科目名を予測 ────────────────────── #
        next_mlm_logits, _ = self.account_encoder.mlm_head(next_hidden + ctx_expanded)

        return {
            "next_mlm_logits": next_mlm_logits,
            "current_account_embeddings": current_ctx,
            "next_account_embeddings": next_account_emb,
        }

    # ------------------------------------------------------------------ #
    # 損失計算
    # ------------------------------------------------------------------ #

    @staticmethod
    def _masked_cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """-100 を無視するcross entropy（有効ラベルが無い場合は0を返す）。"""
        logits_flat = torch.clamp(logits.reshape(-1, logits.size(-1)), -LOGIT_CLAMP, LOGIT_CLAMP)
        labels_flat = labels.reshape(-1)
        valid = labels_flat != -100

        if valid.sum() == 0:
            return torch.zeros((), device=device, requires_grad=True)

        return F.cross_entropy(
            logits_flat[valid], labels_flat[valid], reduction="mean", label_smoothing=0.1
        )

    @staticmethod
    def _clip_labels_to_vocab(labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """語彙サイズを超えるラベルを -100 にして損失計算から除外する。"""
        valid_labels = labels[labels != -100]
        if len(valid_labels) > 0 and valid_labels.max().item() >= vocab_size:
            logger.warning(
                f"ラベルの値が語彙サイズを超えています: {valid_labels.max().item()} >= {vocab_size}"
            )
            labels = labels.clone()
            labels[labels >= vocab_size] = -100
        return labels

    def _compute_current_period_losses(
        self, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """当期のMLM損失・変化クラス損失・変化クラス精度を計算する。"""
        outputs = self(batch)
        account_name_logits = outputs["account_name_logits"]
        change_class_logits = outputs["change_class_logits"]
        device = account_name_logits.device

        # 勘定科目名のMLM損失
        account_name_labels = self._clip_labels_to_vocab(
            batch["account_name_labels"], account_name_logits.size(-1)
        )
        mlm_loss = self._masked_cross_entropy(account_name_logits, account_name_labels, device)

        # 当期の変化クラス分類損失
        valid_change_mask = batch["account_mask"] & (batch["change_class_labels"] != -100)
        if valid_change_mask.sum() == 0:
            return mlm_loss, torch.zeros((), device=device, requires_grad=True), torch.zeros((), device=device)

        predictions = torch.clamp(
            change_class_logits[valid_change_mask], -LOGIT_CLAMP, LOGIT_CLAMP
        )
        labels = batch["change_class_labels"][valid_change_mask]
        change_loss = F.cross_entropy(predictions, labels, reduction="mean", label_smoothing=0.1)
        change_accuracy = (predictions.argmax(dim=-1) == labels).float().mean()

        return mlm_loss, change_loss, change_accuracy

    @staticmethod
    def _select_next_period_subbatch(batch: Dict) -> Dict:
        """Create a sub-batch with samples that have next period data"""
        has_next = batch["has_next_period"]
        has_next_list = has_next.tolist()

        def filter_list(key: str) -> Optional[List]:
            """Filter string lists by boolean tensor (cannot be indexed by bool tensor)"""
            raw = batch.get(key)
            if raw is None:
                return None
            return [value for value, keep in zip(raw, has_next_list) if keep]

        tensor_keys = [
            "input_ids", "attention_mask", "change_class_inputs", "account_mask",
            "next_input_ids", "next_attention_mask", "next_account_name_labels",
            "next_account_mask", "next_change_class_labels",
        ]
        list_keys = [
            "calc_parent_keys", "next_calc_parent_keys",
            "account_names_str", "next_account_names_str",
            "account_keys_str", "next_account_keys_str",
        ]

        sub_batch = {key: batch[key][has_next] for key in tensor_keys}
        sub_batch.update({key: filter_list(key) for key in list_keys})
        return sub_batch

    def _compute_next_period_losses(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute next period MLM loss and change class loss"""
        device = self.device
        zero = torch.zeros((), device=device, requires_grad=True)

        if "has_next_period" not in batch or not batch["has_next_period"].any():
            return zero, zero

        next_batch = self._select_next_period_subbatch(batch)
        outputs = self.forward_next_period(next_batch)

        # Next period MLM loss (next period account names are not masked, so it is actually always 0)
        next_mlm_logits = outputs["next_mlm_logits"]
        next_name_labels = self._clip_labels_to_vocab(
            next_batch["next_account_name_labels"], next_mlm_logits.size(-1)
        )
        next_mlm_loss = self._masked_cross_entropy(next_mlm_logits, next_name_labels, device)

        # Next period change class loss (self-correlation learning from current period to next period via cross-attention)
        next_change_logits = self._predict_next_period_change_class(
            outputs["current_account_embeddings"],
            next_batch["account_mask"],
            outputs["next_account_embeddings"],
        )

        valid_mask = next_batch["next_account_mask"] & (next_batch["next_change_class_labels"] != -100)
        if valid_mask.sum() == 0:
            return next_mlm_loss, zero

        predictions = torch.clamp(next_change_logits[valid_mask], -LOGIT_CLAMP, LOGIT_CLAMP)
        labels = next_batch["next_change_class_labels"][valid_mask]
        next_change_loss = F.cross_entropy(
            predictions, labels, reduction="mean", label_smoothing=0.1
        )

        return next_mlm_loss, next_change_loss

    def _shared_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute training / validation common losses"""
        device = self.device
        zero = torch.zeros((), device=device, requires_grad=True)

        mlm_loss, change_loss, change_accuracy = zero, zero, torch.zeros((), device=device)
        if self.config.enable_current_period_in_pretraining:
            mlm_loss, change_loss, change_accuracy = self._compute_current_period_losses(batch)

        next_mlm_loss, next_change_loss = zero, zero
        if self.config.enable_next_period_in_pretraining:
            next_mlm_loss, next_change_loss = self._compute_next_period_losses(batch)

        total_loss = torch.zeros((), device=device, requires_grad=True)
        if self.config.enable_current_period_in_pretraining:
            total_loss = total_loss + mlm_loss + self.config.change_class_loss_weight * change_loss
        if self.config.enable_next_period_in_pretraining:
            total_loss = total_loss + self.config.next_period_mlm_loss_weight * next_change_loss

        return {
            "total_loss": total_loss,
            "mlm_loss": mlm_loss,
            "change_loss": change_loss,
            "change_accuracy": change_accuracy,
            "next_mlm_loss": next_mlm_loss,
            "next_change_class_loss": next_change_loss,
        }

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        losses = self._shared_step(batch)
        total_loss = losses["total_loss"]

        # 発散した損失はスキップする（学習を継続させるため）
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 100:
            logger.warning(f"異常な損失値を検出したためスキップします: {total_loss.item()}")
            return torch.ones((), device=total_loss.device, requires_grad=True)

        self.log("train_mlm_loss", losses["mlm_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_change_loss", losses["change_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_change_accuracy", losses["change_accuracy"], on_step=False, on_epoch=True)
        self.log("train_next_mlm_loss", losses["next_mlm_loss"], on_step=False, on_epoch=True)
        self.log("train_next_change_class_loss", losses["next_change_class_loss"],
                 on_step=False, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        losses = self._shared_step(batch)
        total_loss = losses["total_loss"]

        if torch.isnan(total_loss):
            logger.warning("NaN loss detected in validation")
            return torch.zeros((), device=total_loss.device)

        self.log("val_mlm_loss", losses["mlm_loss"], on_epoch=True, prog_bar=True)
        self.log("val_change_loss", losses["change_loss"], on_epoch=True, prog_bar=True)
        self.log("val_change_accuracy", losses["change_accuracy"], on_epoch=True, prog_bar=True)
        self.log("val_next_mlm_loss", losses["next_mlm_loss"], on_epoch=True)
        self.log("val_next_change_class_loss", losses["next_change_class_loss"], on_epoch=True)
        self.log("val_total_loss", total_loss, on_epoch=True, prog_bar=True)

        return total_loss

    def on_validation_epoch_end(self):
        """Log loss breakdown at the end of validation epoch"""
        metrics = self.trainer.callback_metrics

        logger.info("=" * 80)
        logger.info("Account Pretraining Validation Loss Breakdown (End of Epoch)")
        logger.info("=" * 80)

        total_loss = metrics.get("val_total_loss")
        if total_loss is not None:
            logger.info(f"Total Validation Loss: {total_loss.item():.6f}")
            logger.info("-" * 80)

        formula_parts: List[str] = []
        value_parts: List[str] = []

        if self.config.enable_current_period_in_pretraining:
            mlm_loss = metrics.get("val_mlm_loss")
            change_loss = metrics.get("val_change_loss")
            change_accuracy = metrics.get("val_change_accuracy")
            weight = self.config.change_class_loss_weight

            logger.info("Current Period")
            if mlm_loss is not None:
                logger.info(f"  MLM Loss (Account Name Mask Prediction): {mlm_loss.item():.6f}")
            if change_loss is not None:
                logger.info(
                    f"  Change Class Loss (変化クラス分類): {change_loss.item():.6f} "
                    f"(重み付き: {change_loss.item() * weight:.6f})"
                )
            if change_accuracy is not None:
                logger.info(
                    f"  Change Class Accuracy: {change_accuracy.item():.4f} "
                    f"({change_accuracy.item() * 100:.2f}%)"
                )

            formula_parts += ["mlm_loss", f"{weight}*change_loss"]
            if mlm_loss is not None and change_loss is not None:
                value_parts += [f"{mlm_loss.item():.6f}", f"{change_loss.item() * weight:.6f}"]

        if self.config.enable_next_period_in_pretraining:
            next_change_loss = metrics.get("val_next_change_class_loss")
            weight = self.config.next_period_mlm_loss_weight

            if next_change_loss is not None:
                logger.info("-" * 80)
                logger.info("【翌会計期間】")
                logger.info(
                    f"  Next Change Class Loss (翌期変化クラス自己相関): "
                    f"{next_change_loss.item():.6f} (重み付き: {next_change_loss.item() * weight:.6f})"
                )
                value_parts.append(f"{next_change_loss.item() * weight:.6f}")
            formula_parts.append(f"{weight}*next_change_class_loss")

        if formula_parts:
            logger.info("-" * 80)
            logger.info("損失の構成:")
            logger.info(f"  total_loss = {' + '.join(formula_parts)}")
            if value_parts and total_loss is not None:
                logger.info(f"  {total_loss.item():.6f} = {' + '.join(value_parts)}")

        logger.info("=" * 80)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-8
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss",
                "frequency": 1,
            },
        }


class LightningMultiAccountFinancialCLIP(pl.LightningModule):
    """テキスト情報とのCLIP学習を行う Lightning Module（step2）"""

    def __init__(
        self,
        config: MultiAccountCLIPConfig,
        pretrained_account_encoder: Optional[MultiAccountEncoder] = None,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["pretrained_account_encoder"])

        # --- 勘定科目エンコーダー --------------------------------------- #
        if pretrained_account_encoder is not None:
            self.multi_account_encoder = pretrained_account_encoder
            self._configure_pretrained_encoder()
            logger.info("事前学習済みエンコーダーを統合しました")
        else:
            self.multi_account_encoder = build_account_encoder(
                config, enable_mlm_pretraining=config.use_joint_pretraining_loss
            )

        # --- テキストエンコーダー --------------------------------------- #
        self.text_encoder = TextEncoder(
            model_name=config.text_model,
            output_dim=config.output_dim,
            freeze_bert=config.freeze_text_bert,
            use_lora=config.use_text_lora,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules,
        )

        # 温度パラメータ（logit scale）
        self.temperature = nn.Parameter(
            torch.tensor(np.log(1 / config.temperature)).clamp(min=-5.0, max=5.0)
        )

        self._log_clip_architecture()

    def _configure_pretrained_encoder(self):
        """事前学習済みエンコーダーのMLMヘッドとパラメータを整える。"""
        encoder = self.multi_account_encoder

        if self.config.use_joint_pretraining_loss:
            # Joint optimization ではMLMヘッドを使い続ける
            if not hasattr(encoder, "mlm_head"):
                logger.info("Joint optimization用にMLMヘッドを再作成します")
                encoder.mlm_head = MaskedTokenPrediction(
                    hidden_size=encoder.hidden_size,
                    change_classes=encoder.change_classes,
                    vocab_size=encoder.tokenizer.vocab_size,
                )
            else:
                logger.info("Joint optimization用にMLMヘッドを保持します")
            encoder.enable_mlm_pretraining = True
        else:
            if hasattr(encoder, "mlm_head"):
                delattr(encoder, "mlm_head")
            encoder.enable_mlm_pretraining = False

        # 事前学習で発散したパラメータがあれば再初期化する
        for name, param in encoder.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.warning(f"NaN/Inf detected in pretrained parameter {name}, reinitializing")
                nn.init.xavier_uniform_(param.data, gain=0.1)

    def _log_clip_architecture(self):
        """CLIP学習で選択されているアーキテクチャ構成をログ出力する。"""
        config = self.config

        if config.use_graph_attention:
            arch, detail = "Graph Attention", "GAT encode + readout → projection"
        else:
            arch, detail = "Company Attention（デフォルト）", "MHA + mean → projection"

        logger.info("=" * 80)
        logger.info("📋 CLIP学習 タスク × アーキテクチャ 構成")
        logger.info("=" * 80)
        logger.info(f"  集約アーキテクチャ : {arch}")
        logger.info(f"  金額増減クラス     : {self.multi_account_encoder.change_classes[:-1]}")
        logger.info(f"  テキストCLIP       : 有効  | アカウント側: {detail}")
        logger.info(f"                              テキスト側  : {config.text_model} → projection")
        if config.use_joint_pretraining_loss:
            logger.info(
                f"  Joint Optimization : 有効（MLM weight={config.joint_mlm_loss_weight}, "
                f"Change weight={config.joint_change_class_loss_weight}）"
            )
        logger.info("=" * 80)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        account_names_list: List[List[str]],
        change_classes_list: List[List[str]],
        text_inputs: Dict[str, torch.Tensor],
        calc_parent_keys_list: Optional[List[Optional[List[str]]]] = None,
        account_keys_list: Optional[List[Optional[List[str]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            company_features: [batch_size, output_dim]（L2正規化済み）
            text_features:    [batch_size, output_dim]（L2正規化済み）
        """
        try:
            company_features = self.multi_account_encoder(
                account_names_list, change_classes_list, calc_parent_keys_list, account_keys_list
            )
            text_features = self.text_encoder(text_inputs)
        except RuntimeError as e:
            # CUDA エラーはバッチをスキップして学習を継続する（再現性のない一過性の失敗が多いため）
            if "CUDA error" not in str(e) and "device-side assert" not in str(e):
                raise
            logger.error(f"CUDA error detected, skipping batch: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            device = next(self.parameters()).device
            zeros = torch.zeros(len(account_names_list), self.config.output_dim, device=device)
            return zeros, zeros.clone()

        # NaN/Inf を含む特徴量はゼロに置き換える（in-place にすると autograd が壊れるため再代入する）
        if torch.isnan(company_features).any() or torch.isinf(company_features).any():
            logger.warning("Company features contain NaN/Inf, using zero features")
            company_features = torch.zeros_like(company_features)
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            logger.warning("Text features contain NaN/Inf, using zero features")
            text_features = torch.zeros_like(text_features)

        # 正規化（ゼロ除算を避けるため微小値を加える）
        company_features = F.normalize(company_features + 1e-8, dim=-1)
        text_features = F.normalize(text_features + 1e-8, dim=-1)

        return company_features, text_features

    # ------------------------------------------------------------------ #
    # CLIP損失
    # ------------------------------------------------------------------ #

    def compute_clip_loss(
        self, company_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """対称的なCLIP損失（InfoNCE）"""
        if torch.isnan(company_features).any() or torch.isnan(text_features).any():
            logger.warning("Features contain NaN in CLIP loss computation")
            return torch.zeros((), device=company_features.device, requires_grad=True)

        logit_scale = torch.exp(torch.clamp(self.temperature, max=5.0))
        logits = torch.matmul(company_features, text_features.T) * logit_scale
        logits = torch.clamp(logits, -LOGIT_CLAMP, LOGIT_CLAMP)

        labels = torch.arange(company_features.shape[0], device=company_features.device)

        loss_c2t = F.cross_entropy(logits, labels, label_smoothing=0.1)   # company → text
        loss_t2c = F.cross_entropy(logits.T, labels, label_smoothing=0.1)  # text → company
        total_loss = (loss_c2t + loss_t2c) / 2

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning("NaN/Inf detected in CLIP loss")
            return torch.zeros((), device=company_features.device, requires_grad=True)

        return total_loss

    def compute_clip_loss_per_text_type(
        self,
        company_features: torch.Tensor,
        text_features: torch.Tensor,
        text_types: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, int], List[float]]:
        """text_typeごとに損失を計算して加重平均する（joint optimization）

        Returns:
            total_loss: text_typeごとの損失の加重平均
            type_losses: text_typeごとの損失
            type_sample_counts: text_typeごとのサンプル数
            weights: 各text_typeに適用された重み
        """
        # text_typeが無い、または1種類のみの場合は通常の損失計算
        if text_types is None or len(set(text_types)) <= 1:
            return self.compute_clip_loss(company_features, text_features), {}, {}, []

        type_to_indices: Dict[str, List[int]] = {}
        for idx, text_type in enumerate(text_types):
            type_to_indices.setdefault(text_type, []).append(idx)

        type_losses: Dict[str, torch.Tensor] = {}
        type_sample_counts: Dict[str, int] = {}
        valid_losses: List[torch.Tensor] = []
        valid_text_types: List[str] = []

        for text_type, indices in type_to_indices.items():
            if len(indices) < 2:  # CLIP損失には最低2サンプル必要
                logger.debug(
                    f"text_type '{text_type}' のサンプルが少なすぎます"
                    f"（{len(indices)}サンプル）。スキップします。"
                )
                continue

            indices_tensor = torch.tensor(indices, device=company_features.device)
            type_loss = self.compute_clip_loss(
                company_features[indices_tensor], text_features[indices_tensor]
            )

            if torch.isnan(type_loss) or torch.isinf(type_loss):
                logger.warning(f"text_type '{text_type}' で無効な損失を検出しました。")
                continue

            type_losses[text_type] = type_loss
            type_sample_counts[text_type] = len(indices)
            valid_losses.append(type_loss)
            valid_text_types.append(text_type)

        if not valid_losses:
            logger.warning("すべてのtext_typeで無効な損失。通常の損失計算にフォールバック。")
            return self.compute_clip_loss(company_features, text_features), {}, {}, []

        weights = self._compute_text_type_weights(
            valid_text_types, type_sample_counts, len(text_types)
        )
        total_loss = sum(w * loss for w, loss in zip(weights, valid_losses))

        return total_loss, type_losses, type_sample_counts, weights

    def _compute_text_type_weights(
        self,
        text_types: List[str],
        sample_counts: Dict[str, int],
        batch_size: int,
    ) -> List[float]:
        """text_typeごとの重みを計算する（合計が1.0になるよう正規化）"""
        mode = self.config.text_type_loss_weight_mode
        equal_weights = [1.0 / len(text_types)] * len(text_types)

        if mode == "equal":
            return equal_weights

        if mode == "sample_size":
            # サンプル数に比例した重み
            weights = [sample_counts[tt] / batch_size for tt in text_types]
        elif mode == "inverse_freq":
            # 逆頻度重み（少数派のtext_typeを重視）
            weights = [1.0 / sample_counts[tt] for tt in text_types]
        elif mode == "manual":
            if self.config.text_type_loss_weights is None:
                logger.warning("手動重みが設定されていません。均等重みにフォールバック。")
                return equal_weights
            weights = []
            for tt in text_types:
                if tt not in self.config.text_type_loss_weights:
                    logger.warning(f"text_type '{tt}' の重みが未設定のため1.0を使用します。")
                weights.append(self.config.text_type_loss_weights.get(tt, 1.0))
        else:
            logger.warning(f"不明な重み付けモード '{mode}'。均等重みにフォールバック。")
            return equal_weights

        total_weight = sum(weights)
        if total_weight <= 0:
            return equal_weights
        return [w / total_weight for w in weights]

    # ------------------------------------------------------------------ #
    # Joint optimization（CLIP学習中の事前学習損失）
    # ------------------------------------------------------------------ #

    def _apply_masking(
        self,
        input_ids: torch.Tensor,
        mask_probability: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """トークンにマスキングを適用する（Joint optimization のMLM用）

        Args:
            input_ids: [num_accounts, seq_len]
            mask_probability: マスク率（None のときは config.mask_probability）。
                              0.8以上のときは選ばれた位置を全て[MASK]に置換する
                              （翌期予測で翌期内の情報を参照させないため）。

        Returns:
            masked_input_ids, labels（マスクしていない位置は -100）
        """
        if mask_probability is None:
            mask_probability = self.config.mask_probability

        labels = input_ids.clone()
        tokenizer = self.multi_account_encoder.tokenizer

        maskable_positions = (
            (input_ids != tokenizer.pad_token_id)
            & (input_ids != tokenizer.cls_token_id)
            & (input_ids != tokenizer.sep_token_id)
        )
        # マスク率が高い場合はBERT戦略（80/10/10）ではなく全て[MASK]にする
        use_bert_strategy = mask_probability < 0.8

        for i in range(input_ids.size(0)):
            maskable_indices = torch.where(maskable_positions[i])[0]

            if len(maskable_indices) == 0:
                labels[i].fill_(-100)
                continue

            num_to_mask = max(1, int(len(maskable_indices) * mask_probability))
            mask_indices = torch.randperm(len(maskable_indices))[:num_to_mask]
            positions_to_mask = maskable_indices[mask_indices]

            for pos in positions_to_mask:
                if not use_bert_strategy:
                    input_ids[i, pos] = tokenizer.mask_token_id
                    continue
                rand = torch.rand(1).item()
                if rand < 0.8:
                    input_ids[i, pos] = tokenizer.mask_token_id
                elif rand < 0.9:
                    input_ids[i, pos] = torch.randint(
                        1, min(tokenizer.vocab_size, 30000), (1,)
                    ).item()
                # 残り10%はそのまま残す

            # マスクされていない位置のラベルは -100
            labels[i, ~maskable_positions[i]] = -100
            non_masked = torch.ones_like(input_ids[i], dtype=torch.bool)
            non_masked[positions_to_mask] = False
            labels[i, non_masked & maskable_positions[i]] = -100

        return input_ids, labels

    def _compute_company_pretraining_losses(
        self,
        account_names: List[str],
        change_classes: List[str],
        mask_probability: Optional[float] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """1企業分のMLM損失と変化クラス損失を計算する（Joint optimization用）

        Returns:
            (mlm_loss, change_loss)。計算できなかった場合はそれぞれ None。
        """
        encoder = self.multi_account_encoder
        device = self.device

        account_names_str = [str(name).strip() for name in account_names if name]
        if not account_names_str:
            return None, None

        try:
            tokenized = encoder.tokenizer(
                account_names_str,
                padding=True,
                truncation=True,
                max_length=JOINT_ACCOUNT_NAME_MAX_LENGTH,
                return_tensors="pt",
            )
        except Exception as e:
            logger.debug(f"Tokenization error: {e}")
            return None, None

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        masked_input_ids, labels = self._apply_masking(input_ids.clone(), mask_probability)

        bert_outputs = encoder.bert(input_ids=masked_input_ids, attention_mask=attention_mask)
        hidden_states = bert_outputs.last_hidden_state

        # 変化クラス情報を account_fusion 経由で融合する（CLIP推論と同一経路）
        change_indices = encoder.change_class_indices(
            change_classes, len(account_names_str), device
        )
        change_embeddings = encoder.change_class_embedding(change_indices)
        pooled_cls = hidden_states[:, 0, :]
        fused = encoder.account_fusion(torch.cat([pooled_cls, change_embeddings], dim=-1))
        fusion_delta = (fused - pooled_cls).unsqueeze(1).expand(-1, hidden_states.size(1), -1)

        account_name_logits, change_class_logits = encoder.mlm_head(hidden_states + fusion_delta)

        # MLM損失
        mlm_loss = None
        labels_flat = labels.view(-1)
        valid_token_mask = labels_flat != -100
        if valid_token_mask.sum() > 0:
            logits_flat = torch.clamp(
                account_name_logits.view(-1, account_name_logits.size(-1)),
                -LOGIT_CLAMP, LOGIT_CLAMP,
            )
            mlm_loss = F.cross_entropy(
                logits_flat[valid_token_mask], labels_flat[valid_token_mask],
                reduction="mean", label_smoothing=0.1,
            )

        # 変化クラス損失（各勘定科目の先頭トークン位置の出力を使う）
        change_predictions = torch.clamp(
            change_class_logits[:, 0, :], -LOGIT_CLAMP, LOGIT_CLAMP
        )
        change_loss = F.cross_entropy(change_predictions, change_indices, label_smoothing=0.1)

        return mlm_loss, change_loss

    def _compute_pretraining_losses(
        self,
        account_names_list: List[List[str]],
        change_classes_list: List[List[str]],
        account_names_next_list: Optional[List[Optional[List[str]]]] = None,
        change_classes_next_list: Optional[List[Optional[List[str]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint optimization 用の事前学習損失（MLM + 変化クラス予測）を計算する

        Returns:
            mlm_loss, change_class_loss, next_mlm_loss, next_change_class_loss
        """
        zero = torch.zeros((), device=self.device, requires_grad=True)

        if not self.config.use_joint_pretraining_loss:
            return zero, zero, zero, zero

        mlm_losses: List[torch.Tensor] = []
        change_losses: List[torch.Tensor] = []
        next_mlm_losses: List[torch.Tensor] = []
        next_change_losses: List[torch.Tensor] = []

        batch_size = len(account_names_list)
        account_names_next_list = account_names_next_list or [None] * batch_size
        change_classes_next_list = change_classes_next_list or [None] * batch_size

        for i, (account_names, change_classes) in enumerate(
            zip(account_names_list, change_classes_list)
        ):
            if not account_names:
                continue

            mlm_loss, change_loss = self._compute_company_pretraining_losses(
                account_names, change_classes
            )
            if mlm_loss is not None:
                mlm_losses.append(mlm_loss)
            if change_loss is not None:
                change_losses.append(change_loss)

            if not self.config.use_joint_next_period_loss:
                continue

            names_next = account_names_next_list[i] if i < len(account_names_next_list) else None
            classes_next = change_classes_next_list[i] if i < len(change_classes_next_list) else None
            if names_next is None or classes_next is None:
                continue

            # 翌期は高いマスク率で翌期内の情報を参照させない
            next_mlm_loss, next_change_loss = self._compute_company_pretraining_losses(
                names_next, classes_next,
                mask_probability=self.config.next_period_mask_probability,
            )
            if next_mlm_loss is not None:
                next_mlm_losses.append(next_mlm_loss)
            if next_change_loss is not None:
                next_change_losses.append(next_change_loss)

        def mean_or_zero(losses: List[torch.Tensor]) -> torch.Tensor:
            if not losses:
                return zero
            loss = torch.stack(losses).mean()
            if torch.isnan(loss) or torch.isinf(loss):
                return zero
            return loss

        return (
            mean_or_zero(mlm_losses),
            mean_or_zero(change_losses),
            mean_or_zero(next_mlm_losses),
            mean_or_zero(next_change_losses),
        )

    # ------------------------------------------------------------------ #
    # 学習ステップ
    # ------------------------------------------------------------------ #

    @property
    def _use_joint_loss(self) -> bool:
        """Joint optimization を適用するか（事前学習を行わない場合のみ有効）"""
        return self.config.use_joint_pretraining_loss and not self.config.enable_pretraining

    def _shared_step(
        self, batch: Dict, stage: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """training / validation 共通の forward + 損失計算

        Args:
            stage: "train" または "val"（ログのプレフィックスに使う）

        Returns:
            total_loss, clip_loss, company_features, text_features
        """
        company_features, text_features = self(
            batch["account_names_list"],
            batch["change_classes_list"],
            batch["text_inputs"],
            batch.get("calc_parent_keys_list"),
            batch.get("account_keys_list"),
        )

        on_step = stage == "train"

        # --- 対照損失 ---------------------------------------------------- #
        if self.config.use_text_type_loss and "text_types" in batch:
            clip_loss, type_losses, type_sample_counts, weights = (
                self.compute_clip_loss_per_text_type(
                    company_features, text_features, batch["text_types"]
                )
            )
            for idx, (text_type, type_loss) in enumerate(type_losses.items()):
                self.log(f"{stage}_loss_{text_type}", type_loss, on_step=False, on_epoch=True)
                self.log(f"{stage}_samples_{text_type}",
                         float(type_sample_counts.get(text_type, 0)), on_step=False, on_epoch=True)
                if idx < len(weights):
                    self.log(f"{stage}_weight_{text_type}", weights[idx],
                             on_step=False, on_epoch=True)
                    self.log(f"{stage}_weighted_loss_{text_type}", weights[idx] * type_loss,
                             on_step=False, on_epoch=True)
        else:
            clip_loss = self.compute_clip_loss(company_features, text_features)

        # --- Joint optimization の事前学習損失 ---------------------------- #
        if not self._use_joint_loss:
            total_loss = clip_loss
            if self.config.use_joint_pretraining_loss:
                # 事前学習済みモデルを使う場合はCLIP損失のみを最適化する
                self.log(f"{stage}_clip_loss", clip_loss, on_step=on_step, on_epoch=True)
            return total_loss, clip_loss, company_features, text_features

        mlm_loss, change_class_loss, next_mlm_loss, next_change_class_loss = (
            self._compute_pretraining_losses(
                batch["account_names_list"],
                batch["change_classes_list"],
                batch.get("account_names_next_list"),
                batch.get("change_classes_next_list"),
            )
        )

        total_loss = (
            clip_loss
            + self.config.joint_mlm_loss_weight * mlm_loss
            + self.config.joint_change_class_loss_weight * change_class_loss
        )

        self.log(f"{stage}_clip_loss", clip_loss, on_step=on_step, on_epoch=True)
        self.log(f"{stage}_mlm_loss", mlm_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_change_class_loss", change_class_loss, on_step=False, on_epoch=True)

        if self.config.use_joint_next_period_loss:
            total_loss = total_loss + (
                self.config.joint_next_mlm_loss_weight * next_mlm_loss
                + self.config.joint_next_change_class_loss_weight * next_change_class_loss
            )
            self.log(f"{stage}_next_mlm_loss", next_mlm_loss, on_step=False, on_epoch=True)
            self.log(f"{stage}_next_change_class_loss", next_change_class_loss,
                     on_step=False, on_epoch=True)

        return total_loss, clip_loss, company_features, text_features

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # バッチ内のtext_type分布をログ出力（最初の数ステップのみ）
        if self.config.use_text_type_loss and "text_types" in batch and batch_idx < 9:
            text_type_counts: Dict[str, int] = {}
            for text_type in batch["text_types"]:
                text_type_counts[text_type] = text_type_counts.get(text_type, 0) + 1
            logger.info(f"📊 Batch {batch_idx} text_type分布: {text_type_counts}")

        total_loss, _, _, _ = self._shared_step(batch, stage="train")

        self.log("train_loss", total_loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=len(batch["texts"]))
        self.log("temperature", torch.exp(self.temperature), on_step=True)

        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        total_loss, _, company_features, text_features = self._shared_step(batch, stage="val")

        self.log("val_loss", total_loss, on_epoch=True,
                 prog_bar=True, batch_size=len(batch["texts"]))

        # 検索性能の指標（対角要素が正例）
        similarity_matrix = torch.matmul(company_features, text_features.T)
        batch_size = similarity_matrix.shape[0]

        avg_positive_similarity = torch.diag(similarity_matrix).mean()
        off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        avg_negative_similarity = similarity_matrix[off_diagonal].mean()

        actual_indices = torch.arange(batch_size, device=similarity_matrix.device)
        top1_accuracy = (similarity_matrix.argmax(dim=1) == actual_indices).float().mean()

        self.log("val_positive_sim", avg_positive_similarity, on_epoch=True)
        self.log("val_negative_sim", avg_negative_similarity, on_epoch=True)
        self.log("val_sim_gap", avg_positive_similarity - avg_negative_similarity, on_epoch=True)
        self.log("val_top1_acc", top1_accuracy, on_epoch=True, prog_bar=True)

        return total_loss

    def on_validation_epoch_end(self):
        """検証エポック終了時に損失の内訳をログ出力する"""
        metrics = self.trainer.callback_metrics

        logger.info("=" * 80)
        logger.info("📊 Validation Loss 内訳 (Epoch終了時)")
        logger.info("=" * 80)

        total_loss = metrics.get("val_loss")
        if total_loss is not None:
            logger.info(f"Total Validation Loss: {total_loss.item():.6f}")
            logger.info("-" * 80)

        clip_loss = metrics.get("val_clip_loss")

        if self._use_joint_loss:
            logger.info("Joint Optimization 損失内訳 (事前学習なし):")
            if clip_loss is not None:
                logger.info(f"  CLIP Loss: {clip_loss.item():.6f}")

            for label, key, weight in (
                ("MLM Loss", "val_mlm_loss", self.config.joint_mlm_loss_weight),
                ("Change Class Loss", "val_change_class_loss",
                 self.config.joint_change_class_loss_weight),
            ):
                value = metrics.get(key)
                if value is not None:
                    logger.info(
                        f"  {label}: {value.item():.6f} (重み付き: {value.item() * weight:.6f})"
                    )

            if self.config.use_joint_next_period_loss:
                logger.info("翌会計期間 (Next Period) 損失内訳:")
                for label, key, weight in (
                    ("Next MLM Loss", "val_next_mlm_loss",
                     self.config.joint_next_mlm_loss_weight),
                    ("Next Change Class Loss", "val_next_change_class_loss",
                     self.config.joint_next_change_class_loss_weight),
                ):
                    value = metrics.get(key)
                    if value is not None:
                        logger.info(
                            f"  {label}: {value.item():.6f} (重み付き: {value.item() * weight:.6f})"
                        )
            logger.info("-" * 80)
        elif self.config.enable_pretraining and clip_loss is not None:
            logger.info("CLIP Loss (事前学習済み):")
            logger.info(f"  CLIP Loss: {clip_loss.item():.6f}")
            logger.info("-" * 80)

        # text_type別の内訳
        if self.config.use_text_type_loss:
            reserved = {"val_loss", "val_clip_loss", "val_mlm_loss", "val_change_class_loss"}
            text_types_found = {
                key.replace("val_loss_", "")
                for key in metrics
                if key.startswith("val_loss_") and key not in reserved
            }

            if text_types_found:
                logger.info(
                    f"text_type別 損失内訳 (重み付けモード: {self.config.text_type_loss_weight_mode}):"
                )
                logger.info("-" * 80)
                for text_type in sorted(text_types_found):
                    logger.info(f"[{text_type}]")
                    for label, key in (
                        ("Loss", f"val_loss_{text_type}"),
                        ("Samples", f"val_samples_{text_type}"),
                        ("Weight", f"val_weight_{text_type}"),
                        ("Weighted Loss (寄与度)", f"val_weighted_loss_{text_type}"),
                    ):
                        value = metrics.get(key)
                        if value is None:
                            continue
                        if label == "Samples":
                            logger.info(f"  {label}: {int(value.item())}")
                        else:
                            logger.info(f"  {label}: {value.item():.6f}")
                    logger.info("")

        logger.info("=" * 80)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def extract_embeddings(
        self,
        account_names_list: List[List[str]],
        change_classes_list: List[List[str]],
        texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """推論用の埋め込み抽出"""
        self.eval()

        text_inputs = self.text_encoder.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=min(self.config.max_length, 512),
            return_tensors="pt",
        )

        # トークンIDの範囲チェック
        input_ids = text_inputs["input_ids"]
        vocab_size = self.text_encoder.tokenizer.vocab_size
        if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
            logger.warning(
                f"Invalid token IDs in extract_embeddings. "
                f"Range: [{input_ids.min()}, {input_ids.max()}], vocab_size: {vocab_size}"
            )
            text_inputs["input_ids"] = torch.clamp(input_ids, 0, vocab_size - 1)

        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

        with torch.no_grad():
            company_features, text_features = self(
                account_names_list, change_classes_list, text_inputs
            )

        return company_features.cpu(), text_features.cpu()
