# %%
"""Multimodal DCCAE: Deep Canonical Correlation Analysis with Autoencoders
for Financial Statement Features and Text Features.

DCCAE extends Deep CCA by adding reconstruction loss from autoencoders,
which helps learn more robust representations.

Based on:
- https://github.com/ashawkey/CCA/blob/master/deepcca/dccae_mnist.py
- feature/multimodal_deepcca.py

References:
- Wang et al., "On Deep Multi-View Representation Learning", ICML 2015

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
# DCCAE Model Components
# =============================================================================


def cca_loss(
    H1: torch.Tensor,
    H2: torch.Tensor,
    outdim_size: int,
    use_all_singular_values: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """CCA Loss function for DCCAE.

    Args:
        H1: First representation [batch_size, features]
        H2: Second representation [batch_size, features]
        outdim_size: Output dimension size
        use_all_singular_values: Whether to use all singular values
        device: Torch device

    Returns:
        Negative correlation (loss to minimize)

    """
    r1 = 1e-3
    r2 = 1e-3
    eps = 1e-9

    H1, H2 = H1.t(), H2.t()
    o1 = o2 = H1.size(0)
    m = H1.size(1)

    H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
    H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

    SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
    SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(
        H1bar,
        H1bar.t(),
    ) + r1 * torch.eye(o1, device=device)
    SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(
        H2bar,
        H2bar.t(),
    ) + r2 * torch.eye(o2, device=device)

    # Calculating the root inverse of covariance matrices
    D1, V1 = torch.linalg.eigh(SigmaHat11)
    D2, V2 = torch.linalg.eigh(SigmaHat22)

    # Filter out small eigenvalues for stability
    posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]
    posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]

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

    if use_all_singular_values:
        tmp = torch.matmul(Tval.t(), Tval)
        corr = torch.trace(torch.sqrt(tmp))
    else:
        trace_TT = torch.matmul(Tval.t(), Tval)
        trace_TT = torch.add(
            trace_TT,
            (torch.eye(trace_TT.shape[0]) * r1).to(device),
        )
        U, _ = torch.linalg.eigh(trace_TT)
        U = torch.where(
            eps < U,
            U,
            (torch.ones(U.shape).double() * eps).to(device),
        )
        U = U.topk(outdim_size)[0]
        corr = torch.sum(torch.sqrt(U))

    return -corr


def reconstruction_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Reconstruction loss for autoencoders.

    Args:
        reconstructed: Reconstructed input
        original: Original input
        loss_type: Type of loss ('mse' or 'l1')

    Returns:
        Reconstruction loss

    """
    if loss_type == "mse":
        return F.mse_loss(reconstructed, original)
    if loss_type == "l1":
        return F.l1_loss(reconstructed, original)
    raise ValueError(f"Unknown loss type: {loss_type}")


class MLP(nn.Module):
    """Multi-layer perceptron for encoder/decoder."""

    def __init__(
        self,
        layer_sizes: list[int],
        use_batchnorm: bool = True,
        activation: str = "relu",
        output_activation: bool = False,
    ):
        """Initialize MLP.

        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            use_batchnorm: Whether to use batch normalization
            activation: Activation function ('relu', 'elu', 'sigmoid')
            output_activation: Whether to apply activation to output layer

        """
        super().__init__()

        activation_fn = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
        }.get(activation, nn.ReLU)

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Add activation and batchnorm for all but last layer
            # (or all layers if output_activation is True)
            if i < len(layer_sizes) - 2 or output_activation:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(activation_fn())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class DCCAE(nn.Module):
    """Deep Canonical Correlation Analysis with Autoencoders.

    This model combines CCA loss with reconstruction loss from autoencoders,
    learning representations that are both correlated and informative.
    """

    def __init__(
        self,
        encoder_sizes1: list[int],
        encoder_sizes2: list[int],
        input_size1: int,
        input_size2: int,
        latent_size: int,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ):
        """Initialize DCCAE model.

        Args:
            encoder_sizes1: Hidden layer sizes for encoder 1 (FS)
            encoder_sizes2: Hidden layer sizes for encoder 2 (Text)
            input_size1: Input dimension for view 1
            input_size2: Input dimension for view 2
            latent_size: Latent space dimension
            use_batchnorm: Whether to use batch normalization
            activation: Activation function

        """
        super().__init__()

        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.latent_size = latent_size

        # Encoder 1: input_size1 -> encoder_sizes1 -> latent_size
        enc1_sizes = [input_size1] + encoder_sizes1 + [latent_size]
        self.encoder1 = MLP(
            enc1_sizes,
            use_batchnorm=use_batchnorm,
            activation=activation,
            output_activation=False,
        ).double()

        # Decoder 1: latent_size -> encoder_sizes1 (reversed) -> input_size1
        dec1_sizes = [latent_size] + encoder_sizes1[::-1] + [input_size1]
        self.decoder1 = MLP(
            dec1_sizes,
            use_batchnorm=use_batchnorm,
            activation=activation,
            output_activation=False,
        ).double()

        # Encoder 2: input_size2 -> encoder_sizes2 -> latent_size
        enc2_sizes = [input_size2] + encoder_sizes2 + [latent_size]
        self.encoder2 = MLP(
            enc2_sizes,
            use_batchnorm=use_batchnorm,
            activation=activation,
            output_activation=False,
        ).double()

        # Decoder 2: latent_size -> encoder_sizes2 (reversed) -> input_size2
        dec2_sizes = [latent_size] + encoder_sizes2[::-1] + [input_size2]
        self.decoder2 = MLP(
            dec2_sizes,
            use_batchnorm=use_batchnorm,
            activation=activation,
            output_activation=False,
        ).double()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoders and decoders.

        Args:
            x1: First view input [batch_size, input_size1]
            x2: Second view input [batch_size, input_size2]

        Returns:
            Tuple of (z1, z2, r1, r2):
                z1: Latent representation for view 1
                z2: Latent representation for view 2
                r1: Reconstructed view 1
                r2: Reconstructed view 2

        """
        # Encode
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)

        # Decode
        r1 = self.decoder1(z1)
        r2 = self.decoder2(z2)

        return z1, z2, r1, r2

    def encode(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs to latent space.

        Args:
            x1: First view input
            x2: Second view input

        Returns:
            Tuple of latent representations (z1, z2)

        """
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        return z1, z2


class LinearCCA:
    """Linear CCA for post-processing DCCAE outputs."""

    def __init__(self):
        """Initialize Linear CCA."""
        self.w = [None, None]
        self.m = [None, None]
        self.correlations = None

    def fit(self, H1: np.ndarray, H2: np.ndarray, outdim_size: int) -> None:
        """Fit linear CCA on the outputs.

        Args:
            H1: First view representations [samples, features]
            H2: Second view representations [samples, features]
            outdim_size: Number of output dimensions

        """
        r1 = 1e-4
        r2 = 1e-4

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
        SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1**-0.5)), V1.T)
        SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2**-0.5)), V2.T)

        Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        U, D, V = np.linalg.svd(Tval)
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
        result1 = H1 - self.m[0].reshape([1, -1]).repeat(len(H1), axis=0)
        result1 = np.dot(result1, self.w[0])
        result2 = H2 - self.m[1].reshape([1, -1]).repeat(len(H2), axis=0)
        result2 = np.dot(result2, self.w[1])
        return result1, result2


class DCCAESolver:
    """Solver for training DCCAE with early stopping."""

    def __init__(
        self,
        model: DCCAE,
        outdim_size: int,
        epoch_num: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        recon_weight: float = 0.001,
        device: torch.device = torch.device("cpu"),
        patience: int = 15,
        apply_linear_cca: bool = True,
        verbose: bool = True,
        recon_loss_type: str = "mse",
    ):
        """Initialize the solver.

        Args:
            model: DCCAE model
            outdim_size: Output dimension size
            epoch_num: Maximum number of epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            recon_weight: Weight for reconstruction loss
            device: Torch device
            patience: Early stopping patience
            apply_linear_cca: Whether to apply linear CCA on outputs
            verbose: Whether to print training progress
            recon_loss_type: Type of reconstruction loss ('mse' or 'l1')

        """
        self.model = model.to(device)
        self.outdim_size = outdim_size
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.recon_weight = recon_weight
        self.device = device
        self.patience = patience
        self.verbose = verbose
        self.recon_loss_type = recon_loss_type

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.linear_cca = LinearCCA() if apply_linear_cca else None

        self.train_losses = []
        self.train_cca_losses = []
        self.train_recon_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.best_state_dict = None

    def _compute_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        r1: torch.Tensor,
        r2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss (CCA + reconstruction).

        Args:
            x1, x2: Original inputs
            z1, z2: Latent representations
            r1, r2: Reconstructions

        Returns:
            Tuple of (total_loss, cca_loss, recon_loss)

        """
        cca = cca_loss(
            z1,
            z2,
            self.outdim_size,
            use_all_singular_values=False,
            device=self.device,
        )
        recon1 = reconstruction_loss(r1, x1, self.recon_loss_type)
        recon2 = reconstruction_loss(r2, x2, self.recon_loss_type)
        recon = recon1 + recon2

        total = cca + self.recon_weight * recon

        return total, cca, recon

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

            epoch_losses = []
            epoch_cca_losses = []
            epoch_recon_losses = []

            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()

                batch_x1 = x1_train[batch_idx, :]
                batch_x2 = x2_train[batch_idx, :]

                z1, z2, r1, r2 = self.model(batch_x1, batch_x2)
                loss, cca, recon = self._compute_loss(
                    batch_x1,
                    batch_x2,
                    z1,
                    z2,
                    r1,
                    r2,
                )

                epoch_losses.append(loss.item())
                epoch_cca_losses.append(cca.item())
                epoch_recon_losses.append(recon.item())

                loss.backward()
                self.optimizer.step()

            train_loss = np.mean(epoch_losses)
            train_cca = np.mean(epoch_cca_losses)
            train_recon = np.mean(epoch_recon_losses)

            self.train_losses.append(train_loss)
            self.train_cca_losses.append(train_cca)
            self.train_recon_losses.append(train_recon)

            # Validation
            with torch.no_grad():
                self.model.eval()
                val_loss = self._evaluate_loss(x1_val, x2_val)
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
                        f"loss: {train_loss:.4f} (cca: {train_cca:.4f}, recon: {train_recon:.4f}), "
                        f"val_loss: {val_loss:.4f} (improved)",
                    )
            else:
                patience_counter += 1
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.epoch_num} - "
                        f"loss: {train_loss:.4f} (cca: {train_cca:.4f}, recon: {train_recon:.4f}), "
                        f"val_loss: {val_loss:.4f}",
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
            "train_cca_losses": self.train_cca_losses,
            "train_recon_losses": self.train_recon_losses,
            "val_losses": self.val_losses,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }

    def _evaluate_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Evaluate loss on data."""
        with torch.no_grad():
            data_size = x1.size(0)
            batch_idxs = list(
                BatchSampler(
                    SequentialSampler(range(data_size)),
                    batch_size=self.batch_size,
                    drop_last=False,
                ),
            )

            losses = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                z1, z2, r1, r2 = self.model(batch_x1, batch_x2)
                loss, _, _ = self._compute_loss(batch_x1, batch_x2, z1, z2, r1, r2)
                losses.append(loss.item())

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
                z1, z2, r1, r2 = self.model(batch_x1, batch_x2)
                outputs1.append(z1)
                outputs2.append(z2)
                loss, _, _ = self._compute_loss(batch_x1, batch_x2, z1, z2, r1, r2)
                losses.append(loss.item())

            outputs = [
                torch.cat(outputs1, dim=0).cpu().numpy(),
                torch.cat(outputs2, dim=0).cpu().numpy(),
            ]
            return losses, outputs

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


# %%
# =============================================================================
# DCCAE Application Functions
# =============================================================================


def apply_dccae(
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
    weight_decay: float = 1e-4,
    recon_weight: float = 0.001,
    patience: int = 15,
    val_ratio: float = 0.1,
    random_state: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    DCCAESolver,
    StandardScaler | None,
    StandardScaler | None,
    dict[str, Any],
]:
    """DCCAEを適用して共通空間に投影する

    Args:
        fs_train: 訓練用財務諸表特徴量
        text_train: 訓練用テキスト特徴量
        fs_test: テスト用財務諸表特徴量
        text_test: テスト用テキスト特徴量
        n_components: 潜在空間の次元数
        standardize: 標準化するかどうか
        hidden_layers_fs: FS用エンコーダの隠れ層サイズ
        hidden_layers_text: Text用エンコーダの隠れ層サイズ
        epoch_num: 最大エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        weight_decay: 重み減衰（L2正則化）
        recon_weight: 再構成損失の重み
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

    # 標準化
    fs_scaler = StandardScaler() if standardize else None
    text_scaler = StandardScaler() if standardize else None

    if standardize:
        if verbose:
            print("\nStandardizing features...")
        fs_train_scaled = fs_scaler.fit_transform(fs_train)
        text_train_scaled = text_scaler.fit_transform(text_train)
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
        print("\nCreating DCCAE model...")
        print(f"  FS encoder: {input_size1} -> {hidden_layers_fs} -> {n_components}")
        print(
            f"  FS decoder: {n_components} -> {hidden_layers_fs[::-1]} -> {input_size1}",
        )
        print(
            f"  Text encoder: {input_size2} -> {hidden_layers_text} -> {n_components}",
        )
        print(
            f"  Text decoder: {n_components} -> {hidden_layers_text[::-1]} -> {input_size2}",
        )

    model = DCCAE(
        encoder_sizes1=hidden_layers_fs,
        encoder_sizes2=hidden_layers_text,
        input_size1=input_size1,
        input_size2=input_size2,
        latent_size=n_components,
        use_batchnorm=True,
        activation="relu",
    )

    # ソルバーの作成
    solver = DCCAESolver(
        model=model,
        outdim_size=n_components,
        epoch_num=epoch_num,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        recon_weight=recon_weight,
        device=torch_device,
        patience=patience,
        apply_linear_cca=True,
        verbose=verbose,
    )

    # 訓練
    if verbose:
        print(f"\nTraining DCCAE (max {epoch_num} epochs, patience={patience})...")
        print(f"  Reconstruction weight: {recon_weight}")

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
    """DCCAE投影を結合してDataFrameを作成する"""
    fs_cca_df = pd.DataFrame(
        fs_cca,
        index=docids,
        columns=[f"fs_dccae_{i:02d}" for i in range(n_components)],
    )

    text_cca_df = pd.DataFrame(
        text_cca,
        index=docids,
        columns=[f"text_dccae_{i:02d}" for i in range(n_components)],
    )

    combined = pd.concat([fs_cca_df, text_cca_df], axis=1)
    print(f"\nCombined DCCAE features ({split_name}): {combined.shape}")

    return combined


def transform_with_dccae(
    fs_features: pd.DataFrame,
    text_features: pd.DataFrame,
    solver: DCCAESolver,
    fs_scaler: StandardScaler | None,
    text_scaler: StandardScaler | None,
    n_components: int,
    split_name: str = "data",
    fs_train_columns: list[str] | None = None,
    text_train_columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """学習済みDCCAEモデルを使って新しいデータを変換する

    Args:
        fs_features: 財務諸表特徴量
        text_features: テキスト特徴量
        solver: 学習済みDCCAEソルバー
        fs_scaler: FS用スケーラー
        text_scaler: Text用スケーラー
        n_components: 成分数
        split_name: データセット名（ログ用）
        fs_train_columns: 訓練データのFS特徴量カラム
        text_train_columns: 訓練データのText特徴量カラム

    Returns:
        DCCAE変換後の結合特徴量

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

    # DCCAE変換
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


def transform_fs_only_with_dccae(
    fs_features: pd.DataFrame,
    solver: DCCAESolver,
    fs_scaler: StandardScaler | None,
    text_scaler: StandardScaler | None,
    n_components: int,
    split_name: str = "data",
    fs_train_columns: list[str] | None = None,
    text_train_columns: list[str] | None = None,
) -> pd.DataFrame:
    """学習済みDCCAEモデルを使ってFS特徴量のみを変換する（推論用）

    Args:
        fs_features: 財務諸表特徴量
        solver: 学習済みDCCAEソルバー
        fs_scaler: FS用スケーラー
        text_scaler: Text用スケーラー
        n_components: 成分数
        split_name: データセット名（ログ用）
        fs_train_columns: 訓練データのFS特徴量カラム
        text_train_columns: 訓練データのText特徴量カラム

    Returns:
        FS特徴量のDCCAE変換結果のみ

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

    # DCCAE変換（FS側のみ）
    # encoder1を使用してFS側のみを変換
    fs_tensor = torch.tensor(fs_scaled, dtype=torch.float64)

    with torch.no_grad():
        solver.model.eval()
        fs_encoded = solver.model.encoder1(fs_tensor.to(solver.device)).cpu().numpy()

    # Linear CCAを適用（オプション）
    if solver.linear_cca is not None and solver.linear_cca.w[0] is not None:
        # Linear CCAでさらに変換
        fs_cca = fs_encoded - solver.linear_cca.m[0].reshape([1, -1]).repeat(
            len(fs_encoded), axis=0
        )
        fs_cca = np.dot(fs_cca, solver.linear_cca.w[0])
    else:
        fs_cca = fs_encoded

    # FS側のみのDataFrameを作成
    fs_cca_df = pd.DataFrame(
        fs_cca,
        index=fs_aligned.index,
        columns=[f"fs_dccae_{i:02d}" for i in range(n_components)],
    )

    print(f"\nFS-only DCCAE features ({split_name}): {fs_cca_df.shape}")

    return fs_cca_df


# %%
# =============================================================================
# Main Processing Function
# =============================================================================


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(
        description="Multimodal DCCAE: FS features + Text features with Autoencoders",
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
        help="Number of latent space dimensions",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        default=True,
        help="Standardize features before DCCAE",
    )
    parser.add_argument(
        "--hidden_layers_fs",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden layer sizes for FS encoder",
    )
    parser.add_argument(
        "--hidden_layers_text",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden layer sizes for Text encoder",
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
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--recon_weight",
        type=float,
        default=0.001,
        help="Weight for reconstruction loss",
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


def make_feature_multimodal_dccae(
    fs_method: str = "nmf",
    text_method: str = "nmf",
    n_components: int = 32,
    standardize: bool = True,
    hidden_layers_fs: list[int] | None = None,
    hidden_layers_text: list[int] | None = None,
    epoch_num: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    recon_weight: float = 0.001,
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
    """メイン処理：マルチモーダルDCCAE特徴量を作成して保存

    Args:
        fs_method: 財務諸表の次元削減手法
        text_method: テキスト特徴量抽出手法
        n_components: 潜在空間の次元数
        standardize: 標準化するか
        hidden_layers_fs: FS用エンコーダの隠れ層サイズ
        hidden_layers_text: Text用エンコーダの隠れ層サイズ
        epoch_num: 最大エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        weight_decay: 重み減衰
        recon_weight: 再構成損失の重み
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
    print("Multimodal DCCAE Feature Extraction")
    print("=" * 60)
    print(f"FS method: {fs_method}")
    print(f"Text method: {text_method}")
    print(f"Latent dimensions: {n_components}")
    print(f"Standardize: {standardize}")
    print(f"FS encoder hidden layers: {hidden_layers_fs}")
    print(f"Text encoder hidden layers: {hidden_layers_text}")
    print(f"Max epochs: {epoch_num}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Reconstruction weight: {recon_weight}")
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

    # DCCAEの適用
    print("\n--- Applying DCCAE ---")
    (
        combined_train,
        combined_test,
        solver,
        fs_scaler,
        text_scaler,
        history,
    ) = apply_dccae(
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
        recon_weight=recon_weight,
        patience=patience,
        val_ratio=val_ratio,
        random_state=random_state,
        device=device,
        verbose=verbose,
    )

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

            combined_all = transform_fs_only_with_dccae(
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
                    combined_both = transform_with_dccae(
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
                    fs_only_result = transform_fs_only_with_dccae(
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
                        fs_only_result[f"text_dccae_{i:02d}"] = 0.0
                    parts.append(fs_only_result)
            else:
                # textデータなし：全件fsのみで推論
                fs_only_result = transform_fs_only_with_dccae(
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
                    fs_only_result[f"text_dccae_{i:02d}"] = 0.0
                parts.append(fs_only_result)

            if parts:
                combined_fraud = pd.concat(parts).sort_index()

        except FileNotFoundError as e:
            print(f"Warning: Could not load fraud data features: {e}")

    # 保存
    print("\n--- Saving Results ---")

    # パラメータに基づいてサブディレクトリを作成
    dir_parts = ["dccae", fs_method, text_method, f"comp{n_components}"]
    if output_suffix:
        dir_parts.append(output_suffix)

    subdir_name = "_".join(dir_parts)
    output_dir = feature_dir / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    base_name = f"multimodal_dccae_{fs_method}_{text_method}"

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
        "recon_weight": recon_weight,
        "history": history,
        "fs_train_columns": fs_train_columns,
        "text_train_columns": text_train_columns,
    }

    pd.to_pickle(model_dict, model_output_path)
    print(f"DCCAE model: {model_output_path}")

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
    print(f"  FS DCCAE features: {n_components}")
    print(f"  Text DCCAE features: {n_components}")

    # 訓練履歴
    print("\n--- Training History ---")
    print(f"Best epoch: {history['best_epoch'] + 1}")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Final training loss: {history['train_losses'][-1]:.4f}")
    print(f"Final CCA loss: {history['train_cca_losses'][-1]:.4f}")
    print(f"Final reconstruction loss: {history['train_recon_losses'][-1]:.4f}")

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
    results = make_feature_multimodal_dccae(
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
        recon_weight=args.recon_weight,
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
# # 標準DCCAE（全データと訂正後データへの推論を含む）
# results = make_feature_multimodal_dccae()
#
# # 結果の取り出し例
# combined_train = results["combined_train"]
# combined_test = results["combined_test"]
# combined_all = results["combined_all"]
# combined_fraud = results["combined_fraud"]
# solver = results["solver"]
# history = results["history"]
#
# # カスタム設定の例
# results_custom = make_feature_multimodal_dccae(
#     hidden_layers_fs=[1024, 512, 256],
#     hidden_layers_text=[512, 256],
#     n_components=64,
#     recon_weight=0.01,  # 再構成損失の重みを増加
#     epoch_num=200,
#     patience=20,
# )
#
# # GPUを使用する例
# results_gpu = make_feature_multimodal_dccae(
#     device="cuda",
#     batch_size=512,
# )
#
# # 訓練データのみ処理する例（高速）
# results_train_only = make_feature_multimodal_dccae(
#     process_all_data=False,
#     process_fraud_data=False,
# )

# %%
