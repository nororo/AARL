# %%
"""Multimodal Deep CCA: Deep Canonical Correlation Analysis for
Financial Statement Features and Text Features.

Based on:
- ref/b20_DeepCCA_torch.py (Deep CCA implementation)
- feature/multimodal_cca.py (feature loading and processing)

References:
- https://github.com/Michaelvll/DeepCCA
- Original work Copyright (c) 2016 Vahid Noroozi
- Modified work Copyright 2019 Zhanghao Wu

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.linalg import svd as scipy_svd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

# Set default tensor type
torch.set_default_dtype(torch.float64)

DATADIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/data/1_raw")
INTERMEDIATEDIR = Path(
    "/Users/noro/Documents/Projects/t_interpretable_fs/data/2_intermediate",
)


# %%
# =============================================================================
# Data Loading Functions (from multimodal_cca.py)
# =============================================================================


def load_fs_features(
    feature_dir: Path,
    method: str = "nmf",
    split: str = "train",
    n_components: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """財務諸表の次元削減特徴量を読み込む

    Args:
        feature_dir: 特徴量ディレクトリ
        method: 次元削減手法（'nmf' or 'pca'）
        split: データ分割（'train', 'all', 'fraud'）
        n_components: 次元数（サブディレクトリ名に使用）

    Returns:
        財務諸表特徴量のDataFrame, コンポーネントのDataFrame（train時のみ）

    """
    subdir = feature_dir / f"fs_{method}_comp{n_components}"
    filename = subdir / f"fsdata_dim_reduced_{method}_{split}.pkl"
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    fs_features = pd.read_pickle(filename)

    fs_comp = None
    if split == "train":
        comp_file = subdir / f"comp_{method}_train.pkl"
        if comp_file.exists():
            fs_comp = pd.read_pickle(comp_file)

    print(f"Loaded FS features ({method}, {split}): {fs_features.shape}")
    return fs_features, fs_comp


def load_text_features(
    feature_dir: Path,
    split: str = "train",
    method: str = "lda",
    n_topics: int = 256,
    max_features: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """テキスト特徴量を読み込む

    Args:
        feature_dir: 特徴量ディレクトリ
        split: データ分割（'train', 'test', 'all', 'amd'）
        method: テキスト特徴量手法（'lda', 'nmf', or 'tfidf'）
        n_topics: トピック数（サブディレクトリ名に使用、ldaのみ）
        max_features: 最大特徴量数（サブディレクトリ名に使用、ldaのみ）

    Returns:
        テキスト特徴量のDataFrame, コンポーネントのDataFrame（存在する場合）

    """
    if method == "lda":
        subdir = feature_dir / f"text_lda_topics{n_topics}_maxf{max_features}"
        filename = subdir / f"text_{method}_{split}.pkl"
        if not filename.exists():
            filename = feature_dir / f"text_{method}_{split}.pkl"
    elif method == "nmf":
        subdir = feature_dir / f"text_nmf_comp{n_topics}"
        filename = subdir / f"text_{method}_{split}.pkl"
        if not filename.exists():
            filename = feature_dir / f"text_{method}_{split}.pkl"
    else:
        filename = feature_dir / f"text_{method}_{split}.pkl"
        subdir = feature_dir

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    text_features = pd.read_pickle(filename)

    text_comp = None
    comp_file = subdir / f"text_{method}_comp.pkl" if subdir.exists() else None
    if comp_file is None or not comp_file.exists():
        comp_file = feature_dir / f"text_{method}_comp.pkl"
    if comp_file.exists():
        text_comp = pd.read_pickle(comp_file)

    print(f"Loaded Text features ({method}, {split}): {text_features.shape}")
    return text_features, text_comp


def align_features(
    fs_features: pd.DataFrame,
    text_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2つの特徴量のdocidを揃える

    Args:
        fs_features: 財務諸表特徴量
        text_features: テキスト特徴量

    Returns:
        揃えられた財務諸表特徴量とテキスト特徴量

    """
    common_docids = fs_features.index.intersection(text_features.index)
    print(f"Common docids: {len(common_docids)}")
    print(f"  FS only: {len(fs_features.index) - len(common_docids)}")
    print(f"  Text only: {len(text_features.index) - len(common_docids)}")

    fs_aligned = fs_features.loc[common_docids].sort_index()
    text_aligned = text_features.loc[common_docids].sort_index()

    return fs_aligned, text_aligned


# %%
# =============================================================================
# Deep CCA Model Components
# =============================================================================


class CCALoss:
    """CCA Loss function for Deep CCA.

    This computes the negative sum of correlations between the outputs of two networks.
    """

    def __init__(
        self,
        outdim_size: int,
        use_all_singular_values: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize CCA loss.

        Args:
            outdim_size: Output dimension size (number of CCA components)
            use_all_singular_values: Whether to use all singular values
            device: Torch device

        """
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def loss(self, H1: torch.Tensor, H2: torch.Tensor) -> torch.Tensor:
        """Compute CCA loss between two representations.

        Args:
            H1: First representation [batch_size, features]
            H2: Second representation [batch_size, features]

        Returns:
            Negative correlation (loss to minimize)

        """
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        # Check for NaN/Inf in inputs
        if torch.any(~torch.isfinite(H1)) or torch.any(~torch.isfinite(H2)):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        H1, H2 = H1.t(), H2.t()
        o1 = o2 = H1.size(0)
        m = H1.size(1)

        # Need at least 2 samples for covariance
        if m < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(
            H1bar,
            H1bar.t(),
        ) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(
            H2bar,
            H2bar.t(),
        ) + r2 * torch.eye(o2, device=self.device)

        # Calculating the root inverse of covariance matrices
        try:
            D1, V1 = torch.linalg.eigh(SigmaHat11)
            D2, V2 = torch.linalg.eigh(SigmaHat22)
        except RuntimeError:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Clamp eigenvalues instead of filtering to maintain matrix dimensions
        D1 = torch.clamp(D1, min=eps)
        D2 = torch.clamp(D2, min=eps)

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1**-0.5)),
            V1.t(),
        )
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2**-0.5)),
            V2.t(),
        )

        Tval = torch.matmul(
            torch.matmul(SigmaHat11RootInv, SigmaHat12),
            SigmaHat22RootInv,
        )

        # Check for NaN/Inf in Tval
        if torch.any(~torch.isfinite(Tval)):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if self.use_all_singular_values:
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp + eps))
        else:
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(
                trace_TT,
                (torch.eye(trace_TT.shape[0], device=self.device) * r1),
            )
            try:
                U, _ = torch.linalg.eigh(trace_TT)
            except RuntimeError:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            U = torch.clamp(U, min=eps)
            # Handle case where we have fewer eigenvalues than outdim_size
            k = min(self.outdim_size, U.shape[0])
            U = U.topk(k)[0]
            corr = torch.sum(torch.sqrt(U))

        return -corr


class MlpNet(nn.Module):
    """Multi-layer perceptron network for Deep CCA."""

    def __init__(
        self,
        layer_sizes: list[int],
        input_size: int,
        activation: str = "elu",
        use_batchnorm: bool = True,
    ):
        """Initialize MLP network.

        Args:
            layer_sizes: List of hidden layer sizes (last one is output size)
            input_size: Input dimension
            activation: Activation function ('elu', 'sigmoid', 'relu')
            use_batchnorm: Whether to use batch normalization

        """
        super().__init__()
        layers = []
        all_sizes = [input_size] + layer_sizes

        activation_fn = {
            "elu": nn.ELU,
            "sigmoid": nn.Sigmoid,
            "relu": nn.ReLU,
        }.get(activation, nn.ELU)

        for l_id in range(len(all_sizes) - 1):
            if l_id == len(all_sizes) - 2:
                # Last layer: no activation
                if use_batchnorm:
                    layers.append(
                        nn.Sequential(
                            nn.BatchNorm1d(
                                num_features=all_sizes[l_id],
                                affine=False,
                            ),
                            nn.Linear(all_sizes[l_id], all_sizes[l_id + 1]),
                        ),
                    )
                else:
                    layers.append(nn.Linear(all_sizes[l_id], all_sizes[l_id + 1]))
            # Hidden layers with activation
            elif use_batchnorm:
                layers.append(
                    nn.Sequential(
                        nn.Linear(all_sizes[l_id], all_sizes[l_id + 1]),
                        activation_fn(),
                        nn.BatchNorm1d(
                            num_features=all_sizes[l_id + 1],
                            affine=False,
                        ),
                    ),
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Linear(all_sizes[l_id], all_sizes[l_id + 1]),
                        activation_fn(),
                    ),
                )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x)
        # Clip outputs to prevent extreme values
        x = torch.clamp(x, min=-10.0, max=10.0)
        return x


class DeepCCA(nn.Module):
    """Deep Canonical Correlation Analysis model."""

    def __init__(
        self,
        layer_sizes1: list[int],
        layer_sizes2: list[int],
        input_size1: int,
        input_size2: int,
        outdim_size: int,
        use_all_singular_values: bool = False,
        device: torch.device = torch.device("cpu"),
        activation1: str = "elu",
        activation2: str = "elu",
    ):
        """Initialize Deep CCA model.

        Args:
            layer_sizes1: Layer sizes for first network (FS)
            layer_sizes2: Layer sizes for second network (Text)
            input_size1: Input dimension for first network
            input_size2: Input dimension for second network
            outdim_size: Output dimension (number of CCA components)
            use_all_singular_values: Whether to use all singular values in loss
            device: Torch device
            activation1: Activation function for first network
            activation2: Activation function for second network

        """
        super().__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1, activation1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2, activation2).double()
        self.loss_fn = CCALoss(outdim_size, use_all_singular_values, device)
        self.outdim_size = outdim_size

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both networks.

        Args:
            x1: First view input [batch_size, features1]
            x2: Second view input [batch_size, features2]

        Returns:
            Tuple of transformed representations

        """
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        return output1, output2

    def loss(self, output1: torch.Tensor, output2: torch.Tensor) -> torch.Tensor:
        """Compute CCA loss."""
        return self.loss_fn.loss(output1, output2)


class LinearCCA:
    """Linear CCA for post-processing Deep CCA outputs."""

    def __init__(self):
        """Initialize Linear CCA."""
        self.w = [None, None]
        self.m = [None, None]

    def fit(self, H1: np.ndarray, H2: np.ndarray, outdim_size: int) -> None:
        """Fit linear CCA on the outputs.

        Args:
            H1: First view representations [samples, features]
            H2: Second view representations [samples, features]
            outdim_size: Number of output dimensions

        """
        # Check for NaN/Inf in inputs
        if np.any(~np.isfinite(H1)) or np.any(~np.isfinite(H2)):
            print(
                "Warning: NaN/Inf detected in LinearCCA inputs. Replacing with zeros.",
            )
            H1 = np.nan_to_num(H1, nan=0.0, posinf=0.0, neginf=0.0)
            H2 = np.nan_to_num(H2, nan=0.0, posinf=0.0, neginf=0.0)

        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-10  # Minimum eigenvalue threshold

        m = H1.shape[0]
        o1 = H1.shape[1]
        o2 = H2.shape[1]

        self.m[0] = np.mean(H1, axis=0)
        self.m[1] = np.mean(H2, axis=0)
        H1bar = H1 - np.tile(self.m[0], (m, 1))
        H2bar = H2 - np.tile(self.m[1], (m, 1))

        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o1)
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o2)

        D1, V1 = np.linalg.eigh(SigmaHat11)
        D2, V2 = np.linalg.eigh(SigmaHat22)

        # Clip eigenvalues to avoid numerical instability
        D1 = np.maximum(D1, eps)
        D2 = np.maximum(D2, eps)

        SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1**-0.5)), V1.T)
        SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2**-0.5)), V2.T)

        Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        # Check for NaN/Inf in Tval
        if np.any(~np.isfinite(Tval)):
            print("Warning: NaN/Inf detected in Tval. Replacing with zeros.")
            Tval = np.nan_to_num(Tval, nan=0.0, posinf=0.0, neginf=0.0)

        # Try SVD with fallback options
        try:
            U, D, V = np.linalg.svd(Tval, full_matrices=False)
        except np.linalg.LinAlgError:
            print("Warning: numpy SVD did not converge. Trying scipy SVD...")
            try:
                U, D, V = scipy_svd(Tval, full_matrices=False, lapack_driver="gesvd")
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Warning: scipy SVD also failed: {e}. Using identity fallback.")
                # Fallback: use identity-like matrices
                min_dim = min(o1, o2, outdim_size)
                U = np.eye(o1, min_dim)
                D = np.ones(min_dim)
                V = np.eye(min_dim, o2)

        V = V.T
        self.w[0] = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
        self.w[1] = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
        self.correlations = D[0:outdim_size]

    def transform(
        self,
        H1: np.ndarray,
        H2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted linear CCA.

        Args:
            H1: First view representations
            H2: Second view representations

        Returns:
            Transformed representations

        """
        # Check for NaN/Inf in inputs
        if np.any(~np.isfinite(H1)) or np.any(~np.isfinite(H2)):
            H1 = np.nan_to_num(H1, nan=0.0, posinf=0.0, neginf=0.0)
            H2 = np.nan_to_num(H2, nan=0.0, posinf=0.0, neginf=0.0)

        # Check if weights are valid
        if self.w[0] is None or self.w[1] is None:
            print("Warning: LinearCCA weights not fitted. Returning zeros.")
            return np.zeros(
                (len(H1), self.w[0].shape[1] if self.w[0] is not None else 1),
            ), np.zeros((len(H2), self.w[1].shape[1] if self.w[1] is not None else 1))

        if np.any(~np.isfinite(self.w[0])) or np.any(~np.isfinite(self.w[1])):
            print("Warning: LinearCCA weights contain NaN/Inf. Results may be invalid.")
            self.w[0] = np.nan_to_num(self.w[0], nan=0.0, posinf=0.0, neginf=0.0)
            self.w[1] = np.nan_to_num(self.w[1], nan=0.0, posinf=0.0, neginf=0.0)

        result1 = H1 - self.m[0].reshape([1, -1]).repeat(len(H1), axis=0)
        result1 = np.dot(result1, self.w[0])
        result2 = H2 - self.m[1].reshape([1, -1]).repeat(len(H2), axis=0)
        result2 = np.dot(result2, self.w[1])

        # Check for NaN/Inf in results
        if np.any(~np.isfinite(result1)) or np.any(~np.isfinite(result2)):
            result1 = np.nan_to_num(result1, nan=0.0, posinf=0.0, neginf=0.0)
            result2 = np.nan_to_num(result2, nan=0.0, posinf=0.0, neginf=0.0)

        return result1, result2


class DeepCCASolver:
    """Solver for training Deep CCA with early stopping."""

    def __init__(
        self,
        model: DeepCCA,
        outdim_size: int,
        epoch_num: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: torch.device = torch.device("cpu"),
        patience: int = 10,
        apply_linear_cca: bool = True,
        verbose: bool = True,
    ):
        """Initialize the solver.

        Args:
            model: Deep CCA model
            outdim_size: Output dimension size
            epoch_num: Maximum number of epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            device: Torch device
            patience: Early stopping patience
            apply_linear_cca: Whether to apply linear CCA on outputs
            verbose: Whether to print training progress

        """
        self.model = model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.device = device
        self.patience = patience
        self.verbose = verbose

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.linear_cca = LinearCCA() if apply_linear_cca else None
        self.outdim_size = outdim_size

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.best_state_dict = None

    def fit(
        self,
        x1_train: torch.Tensor,
        x2_train: torch.Tensor,
        x1_val: torch.Tensor,
        x2_val: torch.Tensor,
    ) -> dict[str, Any]:
        """Train the model with early stopping.

        Args:
            x1_train: Training data for first view
            x2_train: Training data for second view
            x1_val: Validation data for first view
            x2_val: Validation data for second view

        Returns:
            Training history dictionary

        """
        x1_train = x1_train.to(self.device)
        x2_train = x2_train.to(self.device)
        x1_val = x1_val.to(self.device)
        x2_val = x2_val.to(self.device)

        data_size = x1_train.size(0)
        patience_counter = 0

        for epoch in range(self.epoch_num):
            # Training
            self.model.train()
            batch_idxs = list(
                BatchSampler(
                    RandomSampler(range(data_size)),
                    batch_size=self.batch_size,
                    drop_last=False,
                ),
            )

            epoch_train_losses = []
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x1 = x1_train[batch_idx, :]
                batch_x2 = x2_train[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                loss = self.model.loss(o1, o2)

                # Check for NaN/Inf loss
                if not torch.isfinite(loss):
                    print(
                        f"Warning: Non-finite loss detected: {loss.item()}. Skipping batch.",
                    )
                    continue

                epoch_train_losses.append(loss.item())
                loss.backward()

                # Gradient clipping for numerical stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

            train_loss = np.mean(epoch_train_losses)
            self.train_losses.append(train_loss)

            # Validation
            with torch.no_grad():
                self.model.eval()
                val_loss = self._compute_loss(x1_val, x2_val)
                self.val_losses.append(val_loss)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
                if self.verbose:
                    print(
                        f"Epoch {epoch + 1}/{self.epoch_num} - "
                        f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f} "
                        f"(improved)",
                    )
            else:
                patience_counter += 1
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.epoch_num} - "
                        f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}",
                    )

                if patience_counter >= self.patience:
                    if self.verbose:
                        print(
                            f"\nEarly stopping at epoch {epoch + 1}. "
                            f"Best epoch: {self.best_epoch + 1}",
                        )
                    break

        # Load best model
        self.model.load_state_dict(self.best_state_dict)
        self.model.to(self.device)

        # Train linear CCA on outputs
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(x1_train, x2_train)
            self.linear_cca.fit(outputs[0], outputs[1], self.outdim_size)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }

    def transform(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        apply_linear_cca: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform data using trained model.

        Args:
            x1: First view data
            x2: Second view data
            apply_linear_cca: Whether to apply linear CCA transformation

        Returns:
            Tuple of (cca_output1, cca_output2, nn_output1, nn_output2)

        """
        with torch.no_grad():
            _, nn_outputs = self._get_outputs(x1.to(self.device), x2.to(self.device))

        if apply_linear_cca and self.linear_cca is not None:
            cca_outputs = self.linear_cca.transform(nn_outputs[0], nn_outputs[1])
            return cca_outputs[0], cca_outputs[1], nn_outputs[0], nn_outputs[1]
        return nn_outputs[0], nn_outputs[1], nn_outputs[0], nn_outputs[1]

    def _compute_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Compute loss on data."""
        with torch.no_grad():
            losses, _ = self._get_outputs(x1, x2)
            return np.mean(losses)

    def _get_outputs(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[list[float], list[np.ndarray]]:
        """Get model outputs for data."""
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(
                BatchSampler(
                    SequentialSampler(range(data_size)),
                    batch_size=self.batch_size,
                    drop_last=False,
                ),
            )

            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.model.loss(o1, o2)
                losses.append(loss.item())

            outputs = [
                torch.cat(outputs1, dim=0).cpu().numpy(),
                torch.cat(outputs2, dim=0).cpu().numpy(),
            ]

            # Check and handle NaN/Inf in outputs
            for i in range(2):
                if np.any(~np.isfinite(outputs[i])):
                    nan_count = np.sum(~np.isfinite(outputs[i]))
                    total_count = outputs[i].size
                    print(
                        f"Warning: {nan_count}/{total_count} NaN/Inf values "
                        f"in NN output {i + 1}. Replacing with zeros.",
                    )
                    outputs[i] = np.nan_to_num(
                        outputs[i],
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

            return losses, outputs


# %%
# =============================================================================
# Deep CCA Application Functions
# =============================================================================


def apply_deepcca(
    fs_train: pd.DataFrame,
    text_train: pd.DataFrame,
    fs_test: pd.DataFrame | None = None,
    text_test: pd.DataFrame | None = None,
    n_components: int = 32,
    standardize: bool = True,
    hidden_layers_fs: list[int] | None = None,
    hidden_layers_text: list[int] | None = None,
    epoch_num: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    val_ratio: float = 0.1,
    random_state: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    DeepCCASolver,
    StandardScaler | None,
    StandardScaler | None,
    dict[str, Any],
]:
    """Deep CCAを適用して共通空間に投影する

    Args:
        fs_train: 訓練用財務諸表特徴量
        text_train: 訓練用テキスト特徴量
        fs_test: テスト用財務諸表特徴量
        text_test: テスト用テキスト特徴量
        n_components: Deep CCAの出力次元数
        standardize: 標準化するかどうか
        hidden_layers_fs: FS用ネットワークの隠れ層サイズ
        hidden_layers_text: Text用ネットワークの隠れ層サイズ
        epoch_num: 最大エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        weight_decay: 重み減衰（L2正則化）
        patience: Early stoppingの忍耐値
        val_ratio: 検証データの割合（デフォルト0.1 = 9:1分割）
        random_state: ランダムシード
        device: 使用デバイス（'cpu' or 'cuda'）
        verbose: 進捗を表示するか

    Returns:
        訓練用結合特徴量、テスト用結合特徴量、ソルバー、
        FS用スケーラー、Text用スケーラー、訓練履歴

    """
    torch_device = torch.device(device)

    # デフォルトのネットワーク構造
    input_size1 = fs_train.shape[1]
    input_size2 = text_train.shape[1]

    if hidden_layers_fs is None:
        hidden_layers_fs = [512, 256]
    if hidden_layers_text is None:
        hidden_layers_text = [512, 256]

    # 出力層を追加
    layer_sizes1 = hidden_layers_fs + [n_components]
    layer_sizes2 = hidden_layers_text + [n_components]

    # 標準化
    fs_scaler = StandardScaler() if standardize else None
    text_scaler = StandardScaler() if standardize else None

    if standardize:
        if verbose:
            print("\nStandardizing features...")
        fs_train_scaled = fs_scaler.fit_transform(fs_train)
        text_train_scaled = text_scaler.fit_transform(text_train)

        # Check for NaN/Inf after standardization
        if np.any(~np.isfinite(fs_train_scaled)):
            nan_count = np.sum(~np.isfinite(fs_train_scaled))
            print(f"Warning: {nan_count} NaN/Inf values in FS features after scaling")
            fs_train_scaled = np.nan_to_num(
                fs_train_scaled,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        if np.any(~np.isfinite(text_train_scaled)):
            nan_count = np.sum(~np.isfinite(text_train_scaled))
            print(f"Warning: {nan_count} NaN/Inf values in Text features after scaling")
            text_train_scaled = np.nan_to_num(
                text_train_scaled,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        if verbose:
            print(
                f"  FS scaled stats: mean={np.mean(fs_train_scaled):.4f}, "
                f"std={np.std(fs_train_scaled):.4f}, "
                f"min={np.min(fs_train_scaled):.4f}, max={np.max(fs_train_scaled):.4f}",
            )
            print(
                f"  Text scaled stats: mean={np.mean(text_train_scaled):.4f}, "
                f"std={np.std(text_train_scaled):.4f}, "
                f"min={np.min(text_train_scaled):.4f}, max={np.max(text_train_scaled):.4f}",
            )

        # 極端な外れ値をクリップしてNNのNaN explosionを防ぐ
        # 標準化後でも±10シグマを超える値は数値的に不安定
        clip_value = 10.0
        fs_max_before = np.abs(fs_train_scaled).max()
        text_max_before = np.abs(text_train_scaled).max()
        if fs_max_before > clip_value or text_max_before > clip_value:
            if verbose:
                print(
                    f"  Clipping outliers to ±{clip_value} sigma "
                    f"(FS max={fs_max_before:.2f}, Text max={text_max_before:.2f})",
                )
            fs_train_scaled = np.clip(fs_train_scaled, -clip_value, clip_value)
            text_train_scaled = np.clip(text_train_scaled, -clip_value, clip_value)
    else:
        fs_train_scaled = fs_train.to_numpy()
        text_train_scaled = text_train.to_numpy()

    # Train/Validation分割（9:1）
    if verbose:
        print(
            f"\nSplitting training data (train:val = {1 - val_ratio:.0%}:{val_ratio:.0%})...",
        )

    indices = np.arange(len(fs_train_scaled))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=random_state,
    )

    fs_train_split = fs_train_scaled[train_idx]
    fs_val_split = fs_train_scaled[val_idx]
    text_train_split = text_train_scaled[train_idx]
    text_val_split = text_train_scaled[val_idx]

    if verbose:
        print(f"  Training samples: {len(train_idx)}")
        print(f"  Validation samples: {len(val_idx)}")

    # テンソルに変換
    fs_train_tensor = torch.tensor(fs_train_split, dtype=torch.float64)
    fs_val_tensor = torch.tensor(fs_val_split, dtype=torch.float64)
    text_train_tensor = torch.tensor(text_train_split, dtype=torch.float64)
    text_val_tensor = torch.tensor(text_val_split, dtype=torch.float64)

    # モデルの作成
    if verbose:
        print("\nCreating Deep CCA model...")
        print(f"  FS network: {input_size1} -> {layer_sizes1}")
        print(f"  Text network: {input_size2} -> {layer_sizes2}")

    model = DeepCCA(
        layer_sizes1=layer_sizes1,
        layer_sizes2=layer_sizes2,
        input_size1=input_size1,
        input_size2=input_size2,
        outdim_size=n_components,
        use_all_singular_values=False,
        device=torch_device,
    )

    # ソルバーの作成
    solver = DeepCCASolver(
        model=model,
        outdim_size=n_components,
        epoch_num=epoch_num,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=torch_device,
        patience=patience,
        apply_linear_cca=True,
        verbose=verbose,
    )

    # 訓練
    if verbose:
        print(f"\nTraining Deep CCA (max {epoch_num} epochs, patience={patience})...")

    history = solver.fit(
        fs_train_tensor,
        text_train_tensor,
        fs_val_tensor,
        text_val_tensor,
    )

    if verbose:
        print(f"\nTraining completed. Best epoch: {history['best_epoch'] + 1}")
        print(f"Best validation loss: {history['best_val_loss']:.4f}")

    # 全訓練データでの変換
    fs_train_full_tensor = torch.tensor(fs_train_scaled, dtype=torch.float64)
    text_train_full_tensor = torch.tensor(text_train_scaled, dtype=torch.float64)

    fs_cca_train, text_cca_train, _, _ = solver.transform(
        fs_train_full_tensor,
        text_train_full_tensor,
        apply_linear_cca=True,
    )

    # 訓練データの結合
    combined_train = _combine_projections(
        fs_cca_train,
        text_cca_train,
        fs_train.index,
        n_components,
        "train",
    )

    # テストデータの処理
    combined_test = None
    if fs_test is not None and text_test is not None:
        if standardize:
            fs_test_scaled = fs_scaler.transform(fs_test)
            text_test_scaled = text_scaler.transform(text_test)
        else:
            fs_test_scaled = fs_test.to_numpy()
            text_test_scaled = text_test.to_numpy()

        # 極端な外れ値をクリップ（学習時と同じ前処理）
        fs_test_scaled = np.clip(fs_test_scaled, -10.0, 10.0)
        text_test_scaled = np.clip(text_test_scaled, -10.0, 10.0)

        fs_test_tensor = torch.tensor(fs_test_scaled, dtype=torch.float64)
        text_test_tensor = torch.tensor(text_test_scaled, dtype=torch.float64)

        fs_cca_test, text_cca_test, _, _ = solver.transform(
            fs_test_tensor,
            text_test_tensor,
            apply_linear_cca=True,
        )

        combined_test = _combine_projections(
            fs_cca_test,
            text_cca_test,
            fs_test.index,
            n_components,
            "test",
        )

    return combined_train, combined_test, solver, fs_scaler, text_scaler, history


def _combine_projections(
    fs_cca: np.ndarray,
    text_cca: np.ndarray,
    docids: pd.Index,
    n_components: int,
    split_name: str,
) -> pd.DataFrame:
    """Deep CCA投影を結合してDataFrameを作成する"""
    fs_cca_df = pd.DataFrame(
        fs_cca,
        index=docids,
        columns=[f"fs_dcca_{i:02d}" for i in range(n_components)],
    )

    text_cca_df = pd.DataFrame(
        text_cca,
        index=docids,
        columns=[f"text_dcca_{i:02d}" for i in range(n_components)],
    )

    combined = pd.concat([fs_cca_df, text_cca_df], axis=1)
    print(f"\nCombined Deep CCA features ({split_name}): {combined.shape}")

    return combined


def _check_feature_quality(features: pd.DataFrame, split_name: str) -> None:
    """特徴量の品質チェック（モード崩壊・NaN検出）"""
    if features is None:
        return
    nan_count = features.isna().sum().sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values in {split_name} features. Model may have issues.")
    zero_std_cols = (features.std() < 1e-10).sum()
    if zero_std_cols > 0:
        print(
            f"WARNING: {zero_std_cols}/{features.shape[1]} columns have near-zero std in "
            f"{split_name} features. Model may have collapsed (mode collapse)."
        )
    else:
        min_std = features.std().min()
        print(f"Feature quality check ({split_name}): OK. Min std={min_std:.6f}")


def transform_with_deepcca(
    fs_features: pd.DataFrame,
    text_features: pd.DataFrame,
    solver: DeepCCASolver,
    fs_scaler: StandardScaler | None,
    text_scaler: StandardScaler | None,
    n_components: int,
    split_name: str = "data",
    fs_train_columns: list[str] | None = None,
    text_train_columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """学習済みDeep CCAモデルを使って新しいデータを変換する

    Args:
        fs_features: 財務諸表特徴量
        text_features: テキスト特徴量
        solver: 学習済みDeep CCAソルバー
        fs_scaler: FS用スケーラー
        text_scaler: Text用スケーラー
        n_components: 成分数
        split_name: データセット名（ログ用）
        fs_train_columns: 訓練データのFS特徴量カラム
        text_train_columns: 訓練データのText特徴量カラム

    Returns:
        Deep CCA変換後の結合特徴量

    """
    fs_aligned, text_aligned = align_features(fs_features, text_features)

    if len(fs_aligned) == 0:
        print(f"Warning: No common docids for {split_name}. Skipping.")
        return None

    # 訓練データとカラムを揃える
    if fs_train_columns is not None:
        missing_cols = set(fs_train_columns) - set(fs_aligned.columns)
        for col in missing_cols:
            fs_aligned[col] = 0
        fs_aligned = fs_aligned.reindex(columns=fs_train_columns, fill_value=0)

    if text_train_columns is not None:
        missing_cols = set(text_train_columns) - set(text_aligned.columns)
        for col in missing_cols:
            text_aligned[col] = 0
        text_aligned = text_aligned.reindex(columns=text_train_columns, fill_value=0)

    # スケーリング
    if fs_scaler is not None:
        fs_scaled = fs_scaler.transform(fs_aligned)
    else:
        fs_scaled = fs_aligned.to_numpy()

    if text_scaler is not None:
        text_scaled = text_scaler.transform(text_aligned)
    else:
        text_scaled = text_aligned.to_numpy()

    # 極端な外れ値をクリップ（学習時と同じ前処理）
    fs_scaled = np.clip(fs_scaled, -10.0, 10.0)
    text_scaled = np.clip(text_scaled, -10.0, 10.0)

    # Deep CCA変換
    fs_tensor = torch.tensor(fs_scaled, dtype=torch.float64)
    text_tensor = torch.tensor(text_scaled, dtype=torch.float64)

    fs_cca, text_cca, _, _ = solver.transform(
        fs_tensor,
        text_tensor,
        apply_linear_cca=True,
    )

    # 結合
    combined = _combine_projections(
        fs_cca,
        text_cca,
        fs_aligned.index,
        n_components,
        split_name,
    )

    return combined


def transform_fs_only_with_deepcca(
    fs_features: pd.DataFrame,
    solver: DeepCCASolver,
    fs_scaler: StandardScaler | None,
    text_scaler: StandardScaler | None,
    n_components: int,
    split_name: str = "data",
    fs_train_columns: list[str] | None = None,
    text_train_columns: list[str] | None = None,
) -> pd.DataFrame:
    """学習済みDeep CCAモデルを使ってFS特徴量のみを変換する（推論用）

    Args:
        fs_features: 財務諸表特徴量
        solver: 学習済みDeep CCAソルバー
        fs_scaler: FS用スケーラー
        text_scaler: Text用スケーラー
        n_components: 成分数
        split_name: データセット名（ログ用）
        fs_train_columns: 訓練データのFS特徴量カラム
        text_train_columns: 訓練データのText特徴量カラム

    Returns:
        FS特徴量のDeep CCA変換結果のみ

    """
    # FSデータの準備
    fs_aligned = fs_features.copy()

    # 訓練データとカラムを揃える
    if fs_train_columns is not None:
        missing_cols = set(fs_train_columns) - set(fs_aligned.columns)
        for col in missing_cols:
            fs_aligned[col] = 0
        fs_aligned = fs_aligned.reindex(columns=fs_train_columns, fill_value=0)

    # スケーリング
    if fs_scaler is not None:
        fs_scaled = fs_scaler.transform(fs_aligned)
    else:
        fs_scaled = fs_aligned.to_numpy()

    # 極端な外れ値をクリップ（学習時と同じ前処理）
    fs_scaled = np.clip(fs_scaled, -10.0, 10.0)

    # Deep CCA変換（FS側のみ）
    # model1エンコーダを使用してFS側のみを変換
    fs_tensor = torch.tensor(fs_scaled, dtype=torch.float64)

    with torch.no_grad():
        solver.model.eval()
        fs_encoded = solver.model.model1(fs_tensor.to(solver.device)).cpu().numpy()

    # NaN/Infの検出と警告
    if not np.all(np.isfinite(fs_encoded)):
        nan_count = np.sum(~np.isfinite(fs_encoded))
        print(
            f"Warning: {nan_count} NaN/Inf values detected in encoder output "
            f"({split_name}). Model may have collapsed. Replacing with zeros."
        )
        fs_encoded = np.nan_to_num(fs_encoded, nan=0.0, posinf=0.0, neginf=0.0)

    # Linear CCAを適用（オプション）
    if solver.linear_cca is not None and solver.linear_cca.w[0] is not None:
        # Linear CCAでさらに変換
        m0 = solver.linear_cca.m[0]
        w0 = solver.linear_cca.w[0]

        # m[0] や w[0] に NaN が含まれる場合の対処
        if not np.all(np.isfinite(m0)):
            print(f"Warning: NaN/Inf in linear_cca.m[0] ({split_name}). Replacing with zeros.")
            m0 = np.nan_to_num(m0, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(w0)):
            print(f"Warning: NaN/Inf in linear_cca.w[0] ({split_name}). Using encoder output directly.")
            fs_cca = fs_encoded
        else:
            fs_cca = fs_encoded - m0.reshape([1, -1]).repeat(len(fs_encoded), axis=0)
            fs_cca = np.dot(fs_cca, w0)
    else:
        fs_cca = fs_encoded

    # 最終出力の NaN チェック
    if not np.all(np.isfinite(fs_cca)):
        nan_count = np.sum(~np.isfinite(fs_cca))
        print(
            f"Warning: {nan_count} NaN/Inf values in final FS-only output "
            f"({split_name}). Replacing with zeros."
        )
        fs_cca = np.nan_to_num(fs_cca, nan=0.0, posinf=0.0, neginf=0.0)

    # FS側のみのDataFrameを作成
    actual_dim = fs_cca.shape[1] if fs_cca.ndim > 1 else 1
    if actual_dim != n_components:
        print(
            f"Warning: fs_cca has {actual_dim} dimensions but n_components={n_components}. "
            f"Adjusting column count to {actual_dim}."
        )
        n_components = actual_dim

    fs_cca_df = pd.DataFrame(
        fs_cca,
        index=fs_aligned.index,
        columns=[f"fs_dcca_{i:02d}" for i in range(n_components)],
    )

    print(f"\nFS-only Deep CCA features ({split_name}): {fs_cca_df.shape}")

    return fs_cca_df


# %%
# =============================================================================
# Main Processing Function
# =============================================================================


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(
        description="Multimodal Deep CCA: FS features + Text features",
    )
    parser.add_argument(
        "--fs_method",
        type=str,
        default="nmf",
        choices=["nmf", "pca"],
        help="Financial statement dimensionality reduction method",
    )
    parser.add_argument(
        "--text_method",
        type=str,
        default="nmf",
        choices=["lda", "nmf", "tfidf"],
        help="Text feature extraction method",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=32,
        help="Number of Deep CCA output components",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        default=True,
        help="Standardize features before Deep CCA",
    )
    parser.add_argument(
        "--hidden_layers_fs",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes for FS network (default: [512, 256])",
    )
    parser.add_argument(
        "--hidden_layers_text",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes for Text network (default: [512, 256])",
    )
    parser.add_argument(
        "--epoch_num",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation data ratio (default: 0.1 for 9:1 split)",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Output file suffix",
    )
    parser.add_argument(
        "--fs_n_components",
        type=int,
        default=256,
        help="Number of FS feature components",
    )
    parser.add_argument(
        "--text_n_topics",
        type=int,
        default=256,
        help="Number of text topics",
    )
    parser.add_argument(
        "--text_max_features",
        type=int,
        default=500,
        help="Max text features (for LDA subdirectory)",
    )
    parser.add_argument(
        "--process_all_data",
        action="store_true",
        default=True,
        help="Process all data for inference",
    )
    parser.add_argument(
        "--no_process_all_data",
        action="store_false",
        dest="process_all_data",
        help="Skip processing all data",
    )
    parser.add_argument(
        "--process_fraud_data",
        action="store_true",
        default=True,
        help="Process fraud (amendment) data for inference",
    )
    parser.add_argument(
        "--no_process_fraud_data",
        action="store_false",
        dest="process_fraud_data",
        help="Skip processing fraud data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def make_feature_multimodal_deepcca(
    fs_method: str = "nmf",
    text_method: str = "nmf",
    n_components: int = 32,
    standardize: bool = True,
    hidden_layers_fs: list[int] | None = None,
    hidden_layers_text: list[int] | None = None,
    epoch_num: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    val_ratio: float = 0.1,
    output_suffix: str = "",
    fs_n_components: int = 256,
    text_n_topics: int = 256,
    text_max_features: int = 500,
    process_all_data: bool = True,
    process_fraud_data: bool = True,
    device: str = "cpu",
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """メイン処理：マルチモーダルDeep CCA特徴量を作成して保存

    Args:
        fs_method: 財務諸表の次元削減手法
        text_method: テキスト特徴量抽出手法
        n_components: Deep CCA出力成分数
        standardize: 標準化するか
        hidden_layers_fs: FS用ネットワークの隠れ層サイズ
        hidden_layers_text: Text用ネットワークの隠れ層サイズ
        epoch_num: 最大エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        weight_decay: 重み減衰
        patience: Early stoppingの忍耐値
        val_ratio: 検証データの割合
        output_suffix: 出力ファイル名サフィックス
        fs_n_components: FS特徴量の次元数
        text_n_topics: テキスト特徴量のトピック数
        text_max_features: テキスト特徴量の最大特徴量数
        process_all_data: 全データへの推論を行うか
        process_fraud_data: 訂正後データへの推論を行うか
        device: 使用デバイス
        random_state: ランダムシード
        verbose: 進捗を表示するか

    Returns:
        結果の辞書

    """
    if hidden_layers_fs is None:
        hidden_layers_fs = [512, 256]
    if hidden_layers_text is None:
        hidden_layers_text = [512, 256]

    print("=" * 60)
    print("Multimodal Deep CCA Feature Extraction")
    print("=" * 60)
    print(f"FS method: {fs_method}")
    print(f"Text method: {text_method}")
    print(f"Deep CCA components: {n_components}")
    print(f"Standardize: {standardize}")
    print(f"FS hidden layers: {hidden_layers_fs}")
    print(f"Text hidden layers: {hidden_layers_text}")
    print(f"Max epochs: {epoch_num}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Patience: {patience}")
    print(
        f"Validation ratio: {val_ratio} (train:val = {1 - val_ratio:.0%}:{val_ratio:.0%})",
    )
    print(f"Device: {device}")
    print(f"Process all data: {process_all_data}")
    print(f"Process fraud data: {process_fraud_data}")

    feature_dir = INTERMEDIATEDIR / "feature"

    # 訓練データの読み込み
    print("\n--- Loading Training Features ---")
    fs_train, fs_comp = load_fs_features(
        feature_dir,
        method=fs_method,
        split="train",
        n_components=fs_n_components,
    )
    text_train, text_comp = load_text_features(
        feature_dir,
        split="train",
        method=text_method,
        n_topics=text_n_topics,
        max_features=text_max_features,
    )

    # docidを揃える
    print("\n--- Aligning Training Features ---")
    fs_train_aligned, text_train_aligned = align_features(fs_train, text_train)

    # 訓練データのカラム順序を保存
    fs_train_columns = fs_train_aligned.columns.tolist()
    text_train_columns = text_train_aligned.columns.tolist()

    # テストデータの確認
    test_file = feature_dir / f"text_{text_method}_test.pkl"
    has_test_data = test_file.exists()

    fs_test_aligned = None
    text_test_aligned = None

    if has_test_data:
        print("\n--- Loading Test Features ---")
        text_test, _ = load_text_features(
            feature_dir,
            split="test",
            method=text_method,
            n_topics=text_n_topics,
            max_features=text_max_features,
        )
        print("\n--- Aligning Test Features ---")
        fs_test_aligned, text_test_aligned = align_features(fs_train, text_test)
        if len(fs_test_aligned) == 0 or len(text_test_aligned) == 0:
            print("Warning: No common docids. Skipping test data.")
            fs_test_aligned = None
            text_test_aligned = None
    else:
        print("\n--- No Test Data Found ---")

    # Deep CCAの適用
    print("\n--- Applying Deep CCA ---")
    (
        combined_train,
        combined_test,
        solver,
        fs_scaler,
        text_scaler,
        history,
    ) = apply_deepcca(
        fs_train=fs_train_aligned,
        text_train=text_train_aligned,
        fs_test=fs_test_aligned,
        text_test=text_test_aligned,
        n_components=n_components,
        standardize=standardize,
        hidden_layers_fs=hidden_layers_fs,
        hidden_layers_text=hidden_layers_text,
        epoch_num=epoch_num,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        val_ratio=val_ratio,
        random_state=random_state,
        device=device,
        verbose=verbose,
    )

    # トレーニング結果の品質チェック（モード崩壊の検出）
    _check_feature_quality(combined_train, "train")

    # 全データへの推論（FWS特徴量のみで推論）
    combined_all = None
    if process_all_data:
        print("\n--- Processing All Data (FS features only) ---")
        try:
            fs_all, _ = load_fs_features(
                feature_dir,
                method=fs_method,
                split="all",
                n_components=fs_n_components,
            )

            combined_all = transform_fs_only_with_deepcca(
                fs_features=fs_all,
                solver=solver,
                fs_scaler=fs_scaler,
                text_scaler=text_scaler,
                n_components=n_components,
                split_name="all",
                fs_train_columns=fs_train_columns,
                text_train_columns=text_train_columns,
            )
        except FileNotFoundError as e:
            print(f"Warning: Could not load all data features: {e}")

    # 訂正後データへの推論
    combined_fraud = None
    if process_fraud_data:
        print("\n--- Processing Fraud (Amendment) Data ---")
        try:
            fs_fraud, _ = load_fs_features(
                feature_dir,
                method=fs_method,
                split="fraud",
                n_components=fs_n_components,
            )

            # textデータの読み込みを試みる（なくてもfsのみで推論）
            text_fraud = None
            try:
                text_fraud, _ = load_text_features(
                    feature_dir,
                    split="amd",
                    method=text_method,
                    n_topics=text_n_topics,
                    max_features=text_max_features,
                )
            except FileNotFoundError as e:
                print(f"Warning: Text features not found for fraud data: {e}")
                print("Falling back to FS-only inference for all fraud data.")

            parts = []

            if text_fraud is not None:
                # textとfsの共通docidは両方で推論
                common_docids = fs_fraud.index.intersection(text_fraud.index)
                fs_only_docids = fs_fraud.index.difference(text_fraud.index)
                print(f"Fraud data with text:    {len(common_docids)}")
                print(f"Fraud data without text: {len(fs_only_docids)}")

                if len(common_docids) > 0:
                    combined_both = transform_with_deepcca(
                        fs_features=fs_fraud.loc[common_docids],
                        text_features=text_fraud.loc[common_docids],
                        solver=solver,
                        fs_scaler=fs_scaler,
                        text_scaler=text_scaler,
                        n_components=n_components,
                        split_name="fraud (with text)",
                        fs_train_columns=fs_train_columns,
                        text_train_columns=text_train_columns,
                    )
                    parts.append(combined_both)

                if len(fs_only_docids) > 0:
                    fs_only_result = transform_fs_only_with_deepcca(
                        fs_features=fs_fraud.loc[fs_only_docids],
                        solver=solver,
                        fs_scaler=fs_scaler,
                        text_scaler=text_scaler,
                        n_components=n_components,
                        split_name="fraud (FS only)",
                        fs_train_columns=fs_train_columns,
                        text_train_columns=text_train_columns,
                    )
                    # text側をゼロ埋めしてshapeを統一
                    for i in range(n_components):
                        fs_only_result[f"text_dcca_{i:02d}"] = 0.0
                    parts.append(fs_only_result)
            else:
                # textデータなし：全件fsのみで推論
                fs_only_result = transform_fs_only_with_deepcca(
                    fs_features=fs_fraud,
                    solver=solver,
                    fs_scaler=fs_scaler,
                    text_scaler=text_scaler,
                    n_components=n_components,
                    split_name="fraud (FS only)",
                    fs_train_columns=fs_train_columns,
                    text_train_columns=text_train_columns,
                )
                # text側をゼロ埋めしてshapeを統一
                for i in range(n_components):
                    fs_only_result[f"text_dcca_{i:02d}"] = 0.0
                parts.append(fs_only_result)

            if parts:
                combined_fraud = pd.concat(parts).sort_index()

        except FileNotFoundError as e:
            print(f"Warning: Could not load fraud data features: {e}")

    # 保存
    print("\n--- Saving Results ---")

    # パラメータに基づいてサブディレクトリを作成
    dir_parts = ["deepcca", fs_method, text_method, f"comp{n_components}"]
    if output_suffix:
        dir_parts.append(output_suffix)

    subdir_name = "_".join(dir_parts)
    output_dir = feature_dir / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    base_name = f"multimodal_deepcca_{fs_method}_{text_method}"

    # 訓練データ保存
    train_output_path = output_dir / f"{base_name}_train.pkl"
    combined_train.to_pickle(train_output_path)
    print(f"Training features: {train_output_path}")

    # テストデータ保存
    if combined_test is not None:
        test_output_path = output_dir / f"{base_name}_test.pkl"
        combined_test.to_pickle(test_output_path)
        print(f"Test features: {test_output_path}")

    # 全データ保存
    if combined_all is not None:
        all_output_path = output_dir / f"{base_name}_all.pkl"
        combined_all.to_pickle(all_output_path)
        print(f"All data features: {all_output_path}")

    # 訂正後データ保存
    if combined_fraud is not None:
        fraud_output_path = output_dir / f"{base_name}_fraud.pkl"
        combined_fraud.to_pickle(fraud_output_path)
        print(f"Fraud data features: {fraud_output_path}")

    # モデルとスケーラーの保存
    model_output_path = output_dir / f"{base_name}_model.pkl"
    model_dict = {
        "solver": solver,
        "fs_scaler": fs_scaler,
        "text_scaler": text_scaler,
        "n_components": n_components,
        "fs_method": fs_method,
        "text_method": text_method,
        "hidden_layers_fs": hidden_layers_fs,
        "hidden_layers_text": hidden_layers_text,
        "history": history,
        "fs_train_columns": fs_train_columns,
        "text_train_columns": text_train_columns,
    }

    pd.to_pickle(model_dict, model_output_path)
    print(f"Deep CCA model: {model_output_path}")

    # 訓練履歴の保存
    history_output_path = output_dir / f"{base_name}_history.pkl"
    pd.to_pickle(history, history_output_path)
    print(f"Training history: {history_output_path}")

    # 元特徴量成分の保存（可視化用）
    comp_output_path = output_dir / f"{base_name}_comp.pkl"
    pd.to_pickle(
        {
            "fs_comp": fs_comp,
            "text_comp": text_comp,
        },
        comp_output_path,
    )
    print(f"Component matrices: {comp_output_path}")

    # 整列された特徴量の保存
    aligned_output_path = output_dir / f"{base_name}_aligned.pkl"
    pd.to_pickle(
        {
            "fs_train_aligned": fs_train_aligned,
            "text_train_aligned": text_train_aligned,
        },
        aligned_output_path,
    )
    print(f"Aligned features: {aligned_output_path}")

    # Linear CCA weights保存
    if solver.linear_cca is not None:
        lcca_output_path = output_dir / f"{base_name}_linear_cca.pkl"
        pd.to_pickle(
            {
                "w": solver.linear_cca.w,
                "m": solver.linear_cca.m,
                "correlations": solver.linear_cca.correlations,
            },
            lcca_output_path,
        )
        print(f"Linear CCA weights: {lcca_output_path}")

    # サンプル表示
    print("\n--- Sample Features (first 5 rows, first 10 columns) ---")
    print(combined_train.iloc[:5, :10])

    # 統計情報
    print("\n--- Feature Statistics ---")
    print(f"Training samples: {len(combined_train)}")
    if combined_test is not None:
        print(f"Test samples: {len(combined_test)}")
    if combined_all is not None:
        print(f"All data samples: {len(combined_all)}")
    if combined_fraud is not None:
        print(f"Fraud data samples: {len(combined_fraud)}")
    print(f"Total features: {combined_train.shape[1]}")
    print(f"  FS Deep CCA features: {n_components}")
    print(f"  Text Deep CCA features: {n_components}")

    # 訓練履歴
    print("\n--- Training History ---")
    print(f"Best epoch: {history['best_epoch'] + 1}")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Final training loss: {history['train_losses'][-1]:.4f}")

    return {
        "combined_train": combined_train,
        "combined_test": combined_test,
        "combined_all": combined_all,
        "combined_fraud": combined_fraud,
        "solver": solver,
        "fs_train_aligned": fs_train_aligned,
        "text_train_aligned": text_train_aligned,
        "fs_comp": fs_comp,
        "text_comp": text_comp,
        "fs_scaler": fs_scaler,
        "text_scaler": text_scaler,
        "history": history,
    }


# %%
def main():
    """コマンドライン実行用のメイン関数"""
    args = parse_args()
    results = make_feature_multimodal_deepcca(
        fs_method=args.fs_method,
        text_method=args.text_method,
        n_components=args.n_components,
        standardize=args.standardize,
        hidden_layers_fs=args.hidden_layers_fs,
        hidden_layers_text=args.hidden_layers_text,
        epoch_num=args.epoch_num,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        val_ratio=args.val_ratio,
        output_suffix=args.output_suffix,
        fs_n_components=args.fs_n_components,
        text_n_topics=args.text_n_topics,
        text_max_features=args.text_max_features,
        process_all_data=args.process_all_data,
        process_fraud_data=args.process_fraud_data,
        device=args.device,
        random_state=args.random_state,
    )
    return results


# %%
if __name__ == "__main__":
    results = main()

# %%
# Jupyter実行例
#
# # 標準Deep CCA（全データと訂正後データへの推論を含む）
# results = make_feature_multimodal_deepcca()
#
# # 結果の取り出し例
# combined_train = results["combined_train"]
# combined_test = results["combined_test"]
# combined_all = results["combined_all"]
# combined_fraud = results["combined_fraud"]
# solver = results["solver"]
# history = results["history"]
#
# # カスタムネットワーク構造の例
# results_custom = make_feature_multimodal_deepcca(
#     hidden_layers_fs=[1024, 512, 256],
#     hidden_layers_text=[512, 256],
#     n_components=64,
#     epoch_num=200,
#     patience=20,
# )
#
# # GPUを使用する例
# results_gpu = make_feature_multimodal_deepcca(
#     device="cuda",
#     batch_size=512,
# )
#
# # 訓練データのみ処理する例（高速）
# results_train_only = make_feature_multimodal_deepcca(
#     process_all_data=False,
#     process_fraud_data=False,
# )

# %%
