"""
データセットとバッチサンプラー

- MaskedAccountDataset      : 勘定科目エンコーダーの事前学習用データセット
- MixedTextTypeBatchSampler : 複数のtext_typeを混在させたバッチを作るサンプラー（joint optimization）
- TextTypeBatchSampler      : text_typeごとにバッチを作るサンプラー（alternating optimization）
- MultiAccountDataset       : CLIP学習用の複数勘定科目データセット
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer

from .config import CHANGE_CLASS_FALLBACK_INDEX, build_change_class_vocab

logger = logging.getLogger(__name__)

# calc_parent_key が欠損している勘定科目に割り当てる既定の親キー
DEFAULT_PARENT_KEY = "top_account"


class MaskedAccountDataset(Dataset):
    """勘定科目エンコーダーの事前学習用データセット（複数勘定科目対応）

    各企業の複数勘定科目を1つのサンプルとして返すことで、勘定科目間の相互参照を可能にする。

    当期のchange class予測:
        マスクした勘定科目のchange classを、マスクされていない他の勘定科目から予測する

    翌期のchange class予測:
        翌期のchange classを、当期の全勘定科目の情報から予測する

    Args:
        companies_data: load_real_dataset が返す企業データのリスト
        tokenizer: 勘定科目名のトークナイザ
        mask_probability: 当期MLMのマスク率
        max_length: 勘定科目名のトークナイズ最大長
        max_accounts: 1企業あたりの最大勘定科目数（超過分はランダムサンプリング）
        whole_account_masking: True で勘定科目名を単位に一括マスク、False でトークン単位
        change_classes: 金額増減クラス（実クラスのリスト）。None なら config のデフォルト
        change_class_mask_token: 増減クラスのマスク用トークン
    """

    def __init__(
        self,
        companies_data: List[Dict],
        tokenizer: AutoTokenizer,
        mask_probability: float = 0.15,
        max_length: int = 128,
        max_accounts: int = 50,
        whole_account_masking: bool = True,
        change_classes: Optional[Sequence[str]] = None,
        change_class_mask_token: Optional[str] = None,
    ):
        self.companies_data = companies_data
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.max_length = max_length
        self.max_accounts = max_accounts
        self.whole_account_masking = whole_account_masking

        # 増減クラス語彙（実クラス + マスクトークン）。MultiAccountEncoder と同じ並びにする。
        self.change_classes, self.change_class_mask_token = build_change_class_vocab(
            change_classes, change_class_mask_token
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.change_classes)}
        self.change_class_mask_idx = self.class_to_idx[self.change_class_mask_token]
        self.num_real_change_classes = len(self.change_classes) - 1
        self.default_change_class_idx = CHANGE_CLASS_FALLBACK_INDEX

        # 特別トークン
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.companies_data)

    def _change_class_index(self, change_class: Any) -> int:
        """変化クラス名を実クラスのインデックスに変換する（未知ラベルはフォールバック）。"""
        return self.class_to_idx.get(str(change_class), self.default_change_class_idx)

    def __getitem__(self, idx: int) -> Dict:
        """企業の全勘定科目を1サンプルとして返す"""
        company_data = self.companies_data[idx]

        account_names = company_data["account_names"]
        change_classes = company_data["change_classes"]
        calc_parent_keys = company_data.get("calc_parent_keys", None)
        account_keys = company_data.get("account_keys", None)

        # 翌会計期間のデータ（存在する場合）
        account_names_next = company_data.get("account_names_next", None)
        change_classes_next = company_data.get("change_classes_next", None)
        calc_parent_keys_next = company_data.get("calc_parent_keys_next", calc_parent_keys)
        account_keys_next = company_data.get("account_keys_next", None)
        has_next_period = account_names_next is not None and change_classes_next is not None

        current_accounts = self._process_accounts(
            account_names, change_classes,
            apply_change_class_mask=True,
            calc_parent_keys=calc_parent_keys,
            account_keys=account_keys,
        )

        if has_next_period:
            next_accounts = self._process_next_period_accounts(
                account_names_next, change_classes_next,
                calc_parent_keys=calc_parent_keys_next,
                account_keys=account_keys_next,
            )
        else:
            next_accounts = self._create_dummy_next_accounts()

        return {
            # 当期のデータ（複数勘定科目）
            "input_ids": current_accounts["input_ids"],                   # [num_accounts, seq_len]
            "attention_mask": current_accounts["attention_mask"],         # [num_accounts, seq_len]
            "account_name_labels": current_accounts["account_name_labels"],  # [num_accounts, seq_len]
            "change_class_labels": current_accounts["change_class_labels"],  # [num_accounts]
            "change_class_inputs": current_accounts["change_class_inputs"],  # [num_accounts]
            "account_mask": current_accounts["account_mask"],             # [num_accounts]
            "num_accounts": current_accounts["num_accounts"],             # スカラー
            "calc_parent_keys": current_accounts["calc_parent_keys"],     # List[str] or None
            "account_names_str": current_accounts["account_names_str"],   # List[str]
            "account_keys_str": current_accounts["account_keys_str"],     # List[str] or None

            # 翌期のデータ（複数勘定科目）
            "next_input_ids": next_accounts["input_ids"],
            "next_attention_mask": next_accounts["attention_mask"],
            "next_account_name_labels": next_accounts["account_name_labels"],
            "next_change_class_labels": next_accounts["change_class_labels"],
            "next_account_mask": next_accounts["account_mask"],
            "next_calc_parent_keys": next_accounts.get("calc_parent_keys", None),
            "next_account_names_str": next_accounts.get("account_names_str", None),
            "next_account_keys_str": next_accounts.get("account_keys_str", None),
            "has_next_period": has_next_period,
        }

    def _select_valid_indices(
        self,
        account_names: List[str],
        change_classes: List[str],
    ) -> List[int]:
        """有効な勘定科目のインデックスを選ぶ（max_accounts超過分はランダムサンプリング）。"""
        valid_indices = [
            i for i, (name, _) in enumerate(zip(account_names, change_classes))
            if isinstance(name, str) and len(name.strip()) > 0
        ]

        if len(valid_indices) > self.max_accounts:
            sampled = torch.randperm(len(valid_indices))[: self.max_accounts].tolist()
            # 元の並び順を維持する（親子グラフ構築のため）
            valid_indices = [valid_indices[i] for i in sorted(sampled)]

        return valid_indices

    def _collect_key_columns(
        self,
        valid_indices: List[int],
        account_names: List[str],
        calc_parent_keys: Optional[List[str]],
        account_keys: Optional[List[str]],
    ) -> Tuple[Optional[List[str]], List[str], Optional[List[str]]]:
        """サンプリング後のインデックスに合わせて親子グラフ用の文字列列を整える。

        いずれも長さ max_accounts になるようパディングする。
        """
        filtered_parent_keys: Optional[List[str]] = None
        if calc_parent_keys is not None:
            filtered_parent_keys = [
                calc_parent_keys[i] if i < len(calc_parent_keys) else DEFAULT_PARENT_KEY
                for i in valid_indices
            ]
            filtered_parent_keys += [DEFAULT_PARENT_KEY] * (
                self.max_accounts - len(filtered_parent_keys)
            )

        filtered_account_keys: Optional[List[str]] = None
        if account_keys is not None:
            filtered_account_keys = [
                account_keys[i] if i < len(account_keys) else "" for i in valid_indices
            ]
            filtered_account_keys += [""] * (self.max_accounts - len(filtered_account_keys))

        filtered_account_names = [
            account_names[i].strip() for i in valid_indices
            if isinstance(account_names[i], str)
        ]
        filtered_account_names += [""] * (self.max_accounts - len(filtered_account_names))

        return filtered_parent_keys, filtered_account_names, filtered_account_keys

    def _process_accounts(
        self,
        account_names: List[str],
        change_classes: List[str],
        apply_change_class_mask: bool = True,
        calc_parent_keys: Optional[List[str]] = None,
        account_keys: Optional[List[str]] = None,
    ) -> Dict:
        """当期の勘定科目群を処理する（勘定科目名MLM + 変化クラスマスク）"""
        valid_indices = self._select_valid_indices(account_names, change_classes)
        filtered_parent_keys, filtered_account_names, filtered_account_keys = (
            self._collect_key_columns(valid_indices, account_names, calc_parent_keys, account_keys)
        )

        input_ids_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []
        account_name_labels_list: List[torch.Tensor] = []
        change_class_labels_list: List[int] = []
        change_class_inputs_list: List[int] = []
        num_valid = 0

        for idx in valid_indices:
            try:
                tokenized = self.tokenizer(
                    account_names[idx].strip(),
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            except Exception as e:
                logger.warning(f"トークナイゼーションエラー: {account_names[idx]} - {e}")
                continue

            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)

            # 勘定科目名のマスキング
            masked_input_ids, account_name_labels = self._apply_account_name_masking(
                input_ids.clone()
            )

            # 変化クラスのマスキング
            change_class_idx = self._change_class_index(change_classes[idx])
            if apply_change_class_mask and torch.rand(1).item() < self.mask_probability:
                # マスクする: ラベルは正解値、入力は[MASK]インデックス
                change_class_label = change_class_idx
                change_class_input = self.change_class_mask_idx
            else:
                # マスクしない: ラベルは-100（損失計算で無視）、入力は正解値
                change_class_label = -100
                change_class_input = change_class_idx

            input_ids_list.append(masked_input_ids)
            attention_mask_list.append(attention_mask)
            account_name_labels_list.append(account_name_labels)
            change_class_labels_list.append(change_class_label)
            change_class_inputs_list.append(change_class_input)
            num_valid += 1

        # max_accounts までパディング
        while len(input_ids_list) < self.max_accounts:
            input_ids_list.append(
                torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
            )
            attention_mask_list.append(torch.zeros(self.max_length, dtype=torch.long))
            account_name_labels_list.append(
                torch.full((self.max_length,), -100, dtype=torch.long)
            )
            change_class_labels_list.append(-100)
            change_class_inputs_list.append(self.change_class_mask_idx)

        account_mask = torch.zeros(self.max_accounts, dtype=torch.bool)
        account_mask[:num_valid] = True

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "account_name_labels": torch.stack(account_name_labels_list),
            "change_class_labels": torch.tensor(change_class_labels_list, dtype=torch.long),
            "change_class_inputs": torch.tensor(change_class_inputs_list, dtype=torch.long),
            "account_mask": account_mask,
            "num_accounts": num_valid,
            "calc_parent_keys": filtered_parent_keys,
            "account_names_str": filtered_account_names,
            "account_keys_str": filtered_account_keys,
        }

    def _process_next_period_accounts(
        self,
        account_names: List[str],
        change_classes: List[str],
        calc_parent_keys: Optional[List[str]] = None,
        account_keys: Optional[List[str]] = None,
    ) -> Dict:
        """翌期の勘定科目群を処理する

        設計方針:
          - 勘定科目名トークンはマスクしない（visible）
            → cross-attention の query として各科目を一意に識別するため
            → これにより "売掛金の自己相関" と "長期借入金の自己相関" を区別して学習できる
          - account_name_labels は全て -100（翌期の MLM 損失は計算しない）
          - change_class は常に予測対象（翌期の金額変化パターンの自己相関を学習）
        """
        valid_indices = self._select_valid_indices(account_names, change_classes)
        filtered_parent_keys, filtered_account_names, filtered_account_keys = (
            self._collect_key_columns(valid_indices, account_names, calc_parent_keys, account_keys)
        )

        input_ids_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []
        account_name_labels_list: List[torch.Tensor] = []
        change_class_labels_list: List[int] = []
        num_valid = 0

        for idx in valid_indices:
            try:
                tokenized = self.tokenizer(
                    account_names[idx].strip(),
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            except Exception as e:
                logger.warning(f"翌期トークナイゼーションエラー: {account_names[idx]} - {e}")
                continue

            input_ids = tokenized["input_ids"].squeeze(0)

            input_ids_list.append(input_ids)
            attention_mask_list.append(tokenized["attention_mask"].squeeze(0))
            # 翌期の勘定科目名はマスクしないため MLM 損失には寄与しない
            account_name_labels_list.append(torch.full_like(input_ids, -100))
            change_class_labels_list.append(self._change_class_index(change_classes[idx]))
            num_valid += 1

        while len(input_ids_list) < self.max_accounts:
            input_ids_list.append(
                torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
            )
            attention_mask_list.append(torch.zeros(self.max_length, dtype=torch.long))
            account_name_labels_list.append(
                torch.full((self.max_length,), -100, dtype=torch.long)
            )
            change_class_labels_list.append(-100)

        account_mask = torch.zeros(self.max_accounts, dtype=torch.bool)
        account_mask[:num_valid] = True

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "account_name_labels": torch.stack(account_name_labels_list),
            "change_class_labels": torch.tensor(change_class_labels_list, dtype=torch.long),
            "account_mask": account_mask,
            "calc_parent_keys": filtered_parent_keys,
            "account_names_str": filtered_account_names,
            "account_keys_str": filtered_account_keys,
        }

    def _create_dummy_next_accounts(self) -> Dict:
        """翌期データがない場合のダミーデータ（全てパディング扱い）"""
        return {
            "input_ids": torch.full(
                (self.max_accounts, self.max_length), self.pad_token_id, dtype=torch.long
            ),
            "attention_mask": torch.zeros(self.max_accounts, self.max_length, dtype=torch.long),
            "account_name_labels": torch.full(
                (self.max_accounts, self.max_length), -100, dtype=torch.long
            ),
            "change_class_labels": torch.full((self.max_accounts,), -100, dtype=torch.long),
            "account_mask": torch.zeros(self.max_accounts, dtype=torch.bool),
        }

    def _apply_account_name_masking(
        self,
        input_ids: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """勘定科目名のマスキングを適用する（当期MLM用）

        whole_account_masking=True の場合:
            mask_ratio の確率で勘定科目名の全トークンをまとめてマスクする
            （Whole Account Name Masking）。
            同一勘定科目名の残存トークンを手がかりに予測するショートカットを防ぐ。

        whole_account_masking=False の場合:
            mask_ratio の割合のトークンをランダムにマスクする従来のトークンレベルマスキング。

        Args:
            input_ids: 入力トークンID [seq_len]
            mask_ratio: マスク率（None の場合は self.mask_probability）

        Returns:
            masked_input_ids, labels（マスクしていない位置は -100）
        """
        labels = input_ids.clone()

        # パディングトークンと特別トークンはマスク対象外
        maskable_positions = (
            (input_ids != self.pad_token_id)
            & (input_ids != self.tokenizer.cls_token_id)
            & (input_ids != self.tokenizer.sep_token_id)
        )
        maskable_indices = torch.where(maskable_positions)[0]

        if len(maskable_indices) == 0:
            labels.fill_(-100)
            return input_ids, labels

        if mask_ratio is None:
            mask_ratio = self.mask_probability

        if self.whole_account_masking:
            # 勘定科目名全体を mask_ratio の確率でマスクする
            if torch.rand(1).item() >= mask_ratio:
                # マスクしない: ラベルを全て -100 にして損失計算から除外
                labels.fill_(-100)
                return input_ids, labels
            positions_to_mask = maskable_indices
        else:
            num_to_mask = max(1, int(len(maskable_indices) * mask_ratio))
            mask_indices = torch.randperm(len(maskable_indices))[:num_to_mask]
            positions_to_mask = maskable_indices[mask_indices]

        # マスキング戦略:
        #   mask_ratio < 0.5 : BERT戦略（80% [MASK] / 10% ランダム / 10% そのまま）
        #   mask_ratio >= 0.5: 全て [MASK]（高マスク率）
        use_bert_strategy = mask_ratio < 0.5

        for pos in positions_to_mask:
            if use_bert_strategy:
                rand = torch.rand(1).item()
                if rand < 0.8:
                    input_ids[pos] = self.mask_token_id
                elif rand < 0.9:
                    input_ids[pos] = torch.randint(
                        1, min(self.tokenizer.vocab_size, 30000), (1,)
                    ).item()
                # 残り10%はそのまま残す
            else:
                input_ids[pos] = self.mask_token_id

        # マスクされていない位置のラベルは -100（損失計算で無視）
        labels[~maskable_positions] = -100
        non_masked_positions = torch.ones_like(input_ids, dtype=torch.bool)
        non_masked_positions[positions_to_mask] = False
        labels[non_masked_positions & maskable_positions] = -100

        return input_ids, labels

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """バッチコレート関数（複数勘定科目対応）

        Returns:
            Dict with tensors of shape:
            - input_ids           : [batch_size, max_accounts, seq_len]
            - attention_mask      : [batch_size, max_accounts, seq_len]
            - account_name_labels : [batch_size, max_accounts, seq_len]
            - change_class_labels : [batch_size, max_accounts]
            - change_class_inputs : [batch_size, max_accounts]
            - account_mask        : [batch_size, max_accounts]
            - num_accounts        : [batch_size]
            - next_*              : 翌期データ（同様の形状）
        """
        def stack(key: str) -> torch.Tensor:
            return torch.stack([item[key] for item in batch])

        change_class_labels = stack("change_class_labels")
        change_class_inputs = stack("change_class_inputs")
        next_change_class_labels = stack("next_change_class_labels")

        # 入力インデックスは [MASK] を含む全語彙、ラベルは実クラスのみが有効
        change_class_inputs = torch.clamp(change_class_inputs, 0, len(self.change_classes) - 1)
        for labels in (change_class_labels, next_change_class_labels):
            valid = labels != -100
            labels[valid] = torch.clamp(labels[valid], 0, self.num_real_change_classes - 1)

        return {
            # 当期のデータ
            "input_ids": stack("input_ids"),
            "attention_mask": stack("attention_mask"),
            "account_name_labels": stack("account_name_labels"),
            "change_class_labels": change_class_labels,
            "change_class_inputs": change_class_inputs,
            "account_mask": stack("account_mask"),
            "num_accounts": torch.tensor([item["num_accounts"] for item in batch], dtype=torch.long),
            # 文字列列はテンソル化できないためリストのまま保持する
            "calc_parent_keys": [item["calc_parent_keys"] for item in batch],
            "account_names_str": [item["account_names_str"] for item in batch],
            "account_keys_str": [item["account_keys_str"] for item in batch],
            # 翌期のデータ
            "next_input_ids": stack("next_input_ids"),
            "next_attention_mask": stack("next_attention_mask"),
            "next_account_name_labels": stack("next_account_name_labels"),
            "next_change_class_labels": next_change_class_labels,
            "next_account_mask": stack("next_account_mask"),
            "next_calc_parent_keys": [item["next_calc_parent_keys"] for item in batch],
            "next_account_names_str": [item["next_account_names_str"] for item in batch],
            "next_account_keys_str": [item["next_account_keys_str"] for item in batch],
            "has_next_period": torch.tensor(
                [item["has_next_period"] for item in batch], dtype=torch.bool
            ),
        }


class MixedTextTypeBatchSampler(Sampler):
    """複数のtext_typeを混在させたバッチを作成するサンプラー（真のjoint optimization用）

    各バッチに全text_typeのサンプルを含めることで、1ステップごとに
    全text_typeの損失を反映した勾配更新を行う。
    """

    def __init__(self, text_types: List[str], docids: List[str], batch_size: int, shuffle: bool = True):
        self.text_types = text_types
        self.docids = docids
        self.batch_size = batch_size
        self.shuffle = shuffle

        # text_typeごとにインデックスをグループ化し、さらにdocidでサブグループ化する
        self.type_to_docid_indices: Dict[str, Dict[str, List[int]]] = {}
        for idx, (text_type, docid) in enumerate(zip(text_types, docids)):
            self.type_to_docid_indices.setdefault(text_type, {}).setdefault(docid, []).append(idx)

        # 各text_typeのサンプル数と割合
        self.type_sample_counts = {
            text_type: sum(len(indices) for indices in docid_dict.values())
            for text_type, docid_dict in self.type_to_docid_indices.items()
        }
        total_samples = sum(self.type_sample_counts.values())
        self.type_ratios = {
            text_type: count / total_samples
            for text_type, count in self.type_sample_counts.items()
        }

        logger.info("MixedTextTypeBatchSampler初期化 (真のjoint optimization):")
        for text_type, docid_dict in self.type_to_docid_indices.items():
            logger.info(
                f"  {text_type}: {self.type_sample_counts[text_type]}サンプル "
                f"({self.type_ratios[text_type]:.1%}), {len(docid_dict)}個のユニークdocid"
            )
        logger.info(f"  → 各バッチ(size={batch_size})に複数text_typeを混在させます")

    def _target_counts_per_batch(self) -> Dict[str, int]:
        """各text_typeがバッチ内で占める目標サンプル数（割合ベース、最低1）"""
        target_counts = {
            text_type: max(1, int(self.batch_size * ratio))
            for text_type, ratio in self.type_ratios.items()
        }

        # 合計が batch_size を超える場合は割合を維持したままスケールダウンする
        total_target = sum(target_counts.values())
        if total_target > self.batch_size:
            scale = self.batch_size / total_target
            target_counts = {
                text_type: max(1, int(count * scale))
                for text_type, count in target_counts.items()
            }

        return target_counts

    def __iter__(self):
        """複数のtext_typeを混在させたバッチを生成する"""
        # 各text_typeの (index, docid) プールを準備
        type_docid_pools: Dict[str, List[Tuple[int, str]]] = {}
        for text_type, docid_dict in self.type_to_docid_indices.items():
            docid_list = list(docid_dict.keys())
            if self.shuffle:
                random.shuffle(docid_list)

            sample_pool: List[Tuple[int, str]] = []
            for docid in docid_list:
                indices = docid_dict[docid].copy()
                if self.shuffle:
                    random.shuffle(indices)
                sample_pool.extend((idx, docid) for idx in indices)

            type_docid_pools[text_type] = sample_pool

        text_type_list = list(type_docid_pools.keys())
        target_counts_per_batch = self._target_counts_per_batch()

        batches: List[List[int]] = []

        # 全text_typeのサンプルが残っている間だけバッチを作る（joint optimizationを保証）
        while all(len(pool) > 0 for pool in type_docid_pools.values()):
            batch_indices: List[int] = []
            used_docids = set()
            batch_text_types: List[str] = []

            for text_type in text_type_list:
                target_count = target_counts_per_batch[text_type]
                type_batch_indices: List[int] = []
                remaining_pool: List[Tuple[int, str]] = []

                # 同一バッチ内でdocidが重複しないように選ぶ
                for idx, docid in type_docid_pools[text_type]:
                    if docid not in used_docids and len(type_batch_indices) < target_count:
                        type_batch_indices.append(idx)
                        used_docids.add(docid)
                    else:
                        remaining_pool.append((idx, docid))

                type_docid_pools[text_type] = remaining_pool

                if type_batch_indices:
                    batch_indices.extend(type_batch_indices)
                    batch_text_types.append(text_type)

            # 全text_typeが揃ったバッチのみ採用する
            if len(batch_text_types) == len(text_type_list) and len(batch_indices) >= 2:
                if self.shuffle:
                    random.shuffle(batch_indices)
                batches.append(batch_indices)
            else:
                break

        skipped = {tt: len(pool) for tt, pool in type_docid_pools.items()}
        total_skipped = sum(skipped.values())
        if total_skipped > 0:
            logger.info(
                f"  ⚠️ 真のjoint optimization保証のため、{total_skipped}サンプルをスキップしました:"
            )
            for text_type, count in skipped.items():
                if count > 0:
                    logger.info(f"    - {text_type}: {count}サンプル")
        logger.info(f"  ✅ 全text_type混在バッチを{len(batches)}個作成しました")

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        """推定バッチ数（最もサンプルの少ないtext_typeがボトルネックになる）"""
        target_counts_per_batch = self._target_counts_per_batch()

        batches_per_type = [
            self.type_sample_counts[text_type] // target_count if target_count > 0 else 0
            for text_type, target_count in target_counts_per_batch.items()
        ]

        return min(batches_per_type) if batches_per_type else 0


class TextTypeBatchSampler(Sampler):
    """text_typeごとにバッチを作成するサンプラー

    各バッチは単一のtext_typeのみで構成され、Gradient Accumulationと組み合わせて
    text_type間のjoint optimizationを実現する（alternating optimization）。

    注意: docid制約は緩和され、同じdocidのサンプルが同一バッチに含まれることがある。
    """

    def __init__(self, text_types: List[str], docids: List[str], batch_size: int, shuffle: bool = True):
        self.text_types = text_types
        self.docids = docids
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.type_to_docid_indices: Dict[str, Dict[str, List[int]]] = {}
        for idx, (text_type, docid) in enumerate(zip(text_types, docids)):
            self.type_to_docid_indices.setdefault(text_type, {}).setdefault(docid, []).append(idx)

        logger.info("TextTypeBatchSampler初期化 (alternating optimization + gradient accumulation):")

        type_batch_counts = {}
        for text_type, docid_dict in self.type_to_docid_indices.items():
            total_samples = sum(len(indices) for indices in docid_dict.values())
            num_full_batches = total_samples // batch_size  # batch_sizeを満たすバッチのみ使う
            type_batch_counts[text_type] = num_full_batches
            logger.info(
                f"  {text_type}: {total_samples}サンプル → {num_full_batches}バッチ"
                f"（batch_size={batch_size}）"
            )

        if type_batch_counts:
            min_batches = min(type_batch_counts.values())
            num_text_types = len(type_batch_counts)
            logger.info(
                f"  → ラウンドロビン: {min_batches}ラウンド × {num_text_types} text_types "
                f"= {min_batches * num_text_types}バッチ"
            )
            for text_type, num_batches in type_batch_counts.items():
                skipped = num_batches - min_batches
                if skipped > 0:
                    logger.info(
                        f"     ⚠️ {text_type}: {skipped}バッチ"
                        f"（{skipped * batch_size}サンプル）をスキップ"
                    )

    def __iter__(self):
        type_batches: Dict[str, List[List[int]]] = {}

        for text_type, docid_dict in self.type_to_docid_indices.items():
            all_samples: List[int] = []
            for indices in docid_dict.values():
                if self.shuffle:
                    indices = indices.copy()
                    random.shuffle(indices)
                all_samples.extend(indices)

            if self.shuffle:
                random.shuffle(all_samples)

            # batch_sizeちょうどのバッチのみ採用する（CLIP損失には十分なサンプル数が必要）
            type_batches[text_type] = [
                all_samples[i:i + self.batch_size]
                for i in range(0, len(all_samples), self.batch_size)
                if len(all_samples[i:i + self.batch_size]) == self.batch_size
            ]

        text_types = list(type_batches.keys())
        batch_counts = [len(type_batches[tt]) for tt in text_types]
        min_batches = min(batch_counts) if batch_counts else 0

        # ラウンドロビンで各text_typeから1バッチずつ取り出す
        # （gradient accumulationと組み合わせるため、この順序はシャッフルしない）
        for round_idx in range(min_batches):
            for text_type in text_types:
                yield type_batches[text_type][round_idx]

    def __len__(self):
        type_batch_counts = [
            sum(len(indices) for indices in docid_dict.values()) // self.batch_size
            for docid_dict in self.type_to_docid_indices.values()
        ]

        if type_batch_counts:
            return min(type_batch_counts) * len(type_batch_counts)
        return 0


class MultiAccountDataset(Dataset):
    """CLIP学習用の複数勘定科目データセット"""

    # テキストのトークナイズ最大長の上限（モデルのposition embeddingに合わせる）
    MAX_TEXT_LENGTH = 512

    def __init__(
        self,
        companies_data: List[Dict],
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        text_types: Optional[List[str]] = None,
        docids: Optional[List[str]] = None,
    ):
        assert len(companies_data) == len(texts), "データ数とテキスト数が一致しません"
        if text_types is not None:
            assert len(companies_data) == len(text_types), "データ数とtext_type数が一致しません"
        if docids is not None:
            assert len(companies_data) == len(docids), "データ数とdocid数が一致しません"

        self.companies_data = companies_data
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_types = text_types
        self.docids = docids

    def __len__(self) -> int:
        return len(self.companies_data)

    def __getitem__(self, idx: int) -> Dict:
        company_data = self.companies_data[idx]
        docid = self.docids[idx] if self.docids is not None else "unknown"

        return {
            "account_names": company_data["account_names"],
            "change_classes": company_data["change_classes"],
            "text": self.texts[idx],
            "text_type": self.text_types[idx] if self.text_types is not None else "unknown",
            "docid": docid,
            "edinet_code": company_data.get("edinet_code", docid),
            "company_idx": idx,
            # Graph Attention用の親子グラフ情報
            "calc_parent_keys": company_data.get("calc_parent_keys", None),
            "account_keys": company_data.get("account_keys", None),
            # Joint optimization の翌期損失用
            "account_names_next": company_data.get("account_names_next", None),
            "change_classes_next": company_data.get("change_classes_next", None),
            "docid_next": company_data.get("docid_next", None),
        }

    @staticmethod
    def _align_name_class_pair(
        names: Any,
        classes: Any,
        context: str,
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """勘定科目名と変化クラスのペアを文字列リストに正規化し、長さを揃える。

        どちらかが欠けている場合は (None, None) を返す。
        """
        if names is None or classes is None:
            return None, None
        if not isinstance(names, (list, tuple)) or not isinstance(classes, (list, tuple)):
            logger.warning(f"{context}: account_names / change_classes がリストではありません")
            return None, None

        names = [str(name) for name in names if name is not None]
        classes = [str(cls) for cls in classes if cls is not None]

        if len(names) != len(classes):
            logger.warning(
                f"{context}: 長さ不一致 account_names={len(names)}, change_classes={len(classes)}"
            )
            min_len = min(len(names), len(classes))
            names, classes = names[:min_len], classes[:min_len]

        return names, classes

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """カスタムコレート関数（翌会計期間データ対応）"""
        account_names_list: List[List[str]] = []
        change_classes_list: List[List[str]] = []
        texts: List[str] = []
        text_types: List[str] = []
        edinet_codes: List[str] = []
        company_indices: List[int] = []
        calc_parent_keys_list: List[Optional[List[str]]] = []
        account_keys_list: List[Optional[List[str]]] = []
        account_names_next_list: List[Optional[List[str]]] = []
        change_classes_next_list: List[Optional[List[str]]] = []

        for i, item in enumerate(batch):
            account_names, change_classes = self._align_name_class_pair(
                item.get("account_names"), item.get("change_classes"), f"batch[{i}] 当期"
            )
            account_names_list.append(account_names or [])
            change_classes_list.append(change_classes or [])

            names_next, classes_next = self._align_name_class_pair(
                item.get("account_names_next"), item.get("change_classes_next"),
                f"batch[{i}] 翌期",
            )
            account_names_next_list.append(names_next)
            change_classes_next_list.append(classes_next)

            calc_parent_keys_list.append(item.get("calc_parent_keys", None))
            account_keys_list.append(item.get("account_keys", None))
            texts.append(str(item.get("text", "")).strip())
            text_types.append(str(item.get("text_type", "unknown")))
            edinet_codes.append(str(item.get("edinet_code", item.get("docid", "unknown"))))
            company_indices.append(int(item.get("company_idx", 0)))

        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=min(self.max_length, self.MAX_TEXT_LENGTH),
            return_tensors="pt",
            add_special_tokens=True,
        )

        # トークンIDの範囲チェック
        input_ids = text_inputs["input_ids"]
        vocab_size = self.tokenizer.vocab_size
        if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
            logger.warning(
                f"Invalid token IDs in collate_fn. "
                f"Range: [{input_ids.min()}, {input_ids.max()}], vocab_size: {vocab_size}"
            )
            text_inputs["input_ids"] = torch.clamp(input_ids, 0, vocab_size - 1)

        return {
            "account_names_list": account_names_list,
            "change_classes_list": change_classes_list,
            "text_inputs": text_inputs,
            "texts": texts,
            "text_types": text_types,
            "edinet_codes": edinet_codes,
            "company_indices": company_indices,
            "calc_parent_keys_list": calc_parent_keys_list,
            "account_keys_list": account_keys_list,
            "account_names_next_list": account_names_next_list,
            "change_classes_next_list": change_classes_next_list,
        }
