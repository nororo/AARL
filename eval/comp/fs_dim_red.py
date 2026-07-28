# %% rec
# example
# python fs_dim_red.py --method nmf --n_components 256
# python fs_dim_red.py --method autoencoder --n_components 256

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("/Users/noro/Documents/Projects/t_interpretable_fs/src")
from libs.downstream_task import get_task_div_pred_fraud_new
from libs.load_dataset import (
    get_all_response_tbl,
    load_bs_data,
    load_bs_data_amd,
    load_pl_data,
    load_pl_data_amd,
)

DATADIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/data/1_raw")
INTERMEDIATEDIR = Path(
    "/Users/noro/Documents/Projects/t_interpretable_fs/data/2_intermediate",
)
XBRL_PROJPATH = r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
XBRL_PROJDIR = Path(XBRL_PROJPATH)
CFGDIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/src")
cfg = yaml.load(open(CFGDIR / "cfg_exp_main.yaml"), Loader=yaml.FullLoader)
# %% data
#

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


class Autoencoder(nn.Module):
    """Autoencoder for dimensionality reduction."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = None):
        """Initialize the autoencoder.

        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of the latent (encoded) space
            hidden_dims: List of hidden layer dimensions. If None, uses default.

        """
        super().__init__()

        if hidden_dims is None:
            # Default architecture: gradually reduce dimensions
            hidden_dims = [
                max(input_dim // 2, latent_dim * 4),
                max(input_dim // 4, latent_dim * 2),
            ]

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ],
            )
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ],
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)


class AutoencoderWrapper:
    """Wrapper class to provide sklearn-like interface for Autoencoder."""

    def __init__(
        self,
        n_components: int = 256,
        hidden_dims: list = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        device: str = None,
        patience: int = 15,
        val_ratio: float = 0.1,
    ):
        """Initialize the AutoencoderWrapper.

        Args:
            n_components: Dimension of the latent space
            hidden_dims: List of hidden layer dimensions
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            random_state: Random seed for reproducibility
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
            patience: Early stopping patience (epochs without improvement)
            val_ratio: Validation data ratio (default 0.1 for 9:1 split)

        """
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.patience = patience
        self.val_ratio = val_ratio

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler = StandardScaler()
        self.input_dim = None
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def fit(self, X):
        """Fit the autoencoder on training data with early stopping.

        Args:
            X: Training data (numpy array or pandas DataFrame)

        Returns:
            self

        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Standardize the data
        X_scaled = self.scaler.fit_transform(X)

        self.input_dim = X.shape[1]

        # Split into train/validation (9:1)
        print(
            f"Splitting training data "
            f"(train:val = {1 - self.val_ratio:.0%}:{self.val_ratio:.0%})...",
        )
        X_train, X_val = train_test_split(
            X_scaled,
            test_size=self.val_ratio,
            random_state=self.random_state,
        )
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")

        # Create model
        self.model = Autoencoder(
            input_dim=self.input_dim,
            latent_dim=self.n_components,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        # Create data loaders
        train_tensor = torch.FloatTensor(X_train)
        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_tensor = torch.FloatTensor(X_val).to(self.device)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Initialize early stopping variables
        best_state_dict = None
        patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # Training loop with early stopping
        print(
            f"Training Autoencoder (max {self.epochs} epochs, patience={self.patience})...",
        )
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)

                # Forward pass
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_reconstructed = self.model(val_tensor)
                val_loss = criterion(val_reconstructed, val_tensor).item()
            self.val_losses.append(val_loss)

            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_state_dict = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}] - "
                    f"train_loss: {avg_train_loss:.6f}, "
                    f"val_loss: {val_loss:.6f} (improved)",
                )
            else:
                patience_counter += 1
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}] - "
                        f"train_loss: {avg_train_loss:.6f}, "
                        f"val_loss: {val_loss:.6f}",
                    )

                if patience_counter >= self.patience:
                    print(
                        f"\nEarly stopping at epoch {epoch + 1}. "
                        f"Best epoch: {self.best_epoch + 1}",
                    )
                    break

        # Load best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.model.to(self.device)

        print(f"\nTraining completed. Best epoch: {self.best_epoch + 1}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        return self

    def transform(self, X):
        """Transform data to latent space.

        Args:
            X: Data to transform (numpy array or pandas DataFrame)

        Returns:
            Transformed data in latent space

        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Standardize using fitted scaler
        X_scaled = self.scaler.transform(X)

        # Transform
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            encoded = self.model.encode(X_tensor)
            return encoded.cpu().numpy()

    def fit_transform(self, X):
        """Fit the autoencoder and transform data.

        Args:
            X: Training data (numpy array or pandas DataFrame)

        Returns:
            Transformed data in latent space

        """
        self.fit(X)
        return self.transform(X)

    @property
    def components_(self):
        """Get the encoder weights (similar to PCA/NMF components).

        Returns:
            Components matrix of shape (n_components, input_dim)

        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Get the last linear layer of the encoder (latent layer)
        self.model.eval()
        with torch.no_grad():
            # For interpretability, we use the encoder's last layer weights
            # This gives us how each latent dimension relates to the previous layer
            # The weights have shape (n_components, prev_hidden_dim)
            last_encoder_layer = None
            for layer in reversed(list(self.model.encoder)):
                if isinstance(layer, nn.Linear):
                    last_encoder_layer = layer
                    break

            # Also get the first encoder layer for input-to-hidden mapping
            first_encoder_layer = None
            for layer in self.model.encoder:
                if isinstance(layer, nn.Linear):
                    first_encoder_layer = layer
                    break

            if first_encoder_layer is not None and last_encoder_layer is not None:
                # Approximate the full transformation by using the first layer
                # weights scaled by the latent layer weights
                # First layer: (hidden_dim, input_dim)
                # Last layer: (n_components, prev_hidden_dim)
                # For simplicity, if encoder has only one layer, return its weights
                if first_encoder_layer == last_encoder_layer:
                    return first_encoder_layer.weight.cpu().numpy()
                # Use pseudo-inverse approach for better interpretability
                # Return the transposed first layer weights, padded/truncated
                first_weights = first_encoder_layer.weight.cpu().numpy()
                # Shape: (hidden_dim, input_dim)
                # We want: (n_components, input_dim)
                if first_weights.shape[0] >= self.n_components:
                    return first_weights[: self.n_components, :]
                # Pad with zeros
                padded = np.zeros((self.n_components, first_weights.shape[1]))
                padded[: first_weights.shape[0], :] = first_weights
                return padded
            # Fallback: return identity-like matrix
            return np.eye(self.n_components, self.input_dim)


def plot_feature_comp(feature_comp: pd.DataFrame, n_components: int):
    coler_increase = "tomato"
    coler_decrease = "darkblue"

    bar_num = 10
    for num in range(n_components):
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(1, 2, 2)
        tmp = feature_comp.loc[:, num].copy()
        tmp = tmp.sort_values(ascending=False)
        tmp = tmp.iloc[:bar_num].sort_values(ascending=True)
        col_mask = tmp.index.str.contains("up")
        col = pd.Series([coler_increase] * bar_num)
        col[~col_mask] = coler_decrease
        # tmp=tmp*((tmp.index.str.contains("up")*1)-0.5)*2
        # tmp.index=transcripter.loc[tmp.index.values,"標準ラベル（日本語）"].str.strip()
        # tmp.index.str.strip()

        ax.barh(range(bar_num), tmp.values, align="center", color=col)
        plt.yticks(range(bar_num), tmp.index.to_list())
        ax.grid(True)
        fig.suptitle(" #" + str(num))
        # fig.savefig(SCRATCH + "/NMF_JPN_barh/comp_" + str(num) + ".pdf", transparent=True)


def create_feature_vector(
    bs_data: pd.DataFrame,
    pl_data: pd.DataFrame,
    min_nonzero_ratio: float = 0,
    pivot_columns: str = "key",
) -> pd.DataFrame:
    """bs_dataとpl_dataから特徴量ベクトルを生成する

    Args:
        bs_data: BSデータ (docid, key, diff_rate_assets等)
        pl_data: PLデータ (docid, key, diff_rate_assets等)
        min_nonzero_ratio: 非ゼロ値の最小割合 (これ以下のカラムは削除)

    Returns:
        特徴量ベクトル (docidをインデックスとしたDataFrame)

    """
    # BSデータをピボット（勘定科目ごとのdiff_rate_assetsを特徴量に）
    bs_pivot = pd.pivot_table(
        data=bs_data,
        index="docid",
        columns=pivot_columns,
        values="diff_rate_assets",
        aggfunc="first",
    ).add_prefix("bs_")

    # PLデータをピボット
    pl_pivot = pd.pivot_table(
        data=pl_data,
        index="docid",
        columns=pivot_columns,
        values="diff_rate_assets",
        aggfunc="first",
    ).add_prefix("pl_")

    # 結合
    features = pd.concat([bs_pivot, pl_pivot], axis=1)
    features = features.fillna(0)

    # ほとんど0のカラムを削除（非ゼロ値の割合が閾値以下のカラム）
    nonzero_ratio = (features != 0).sum() / len(features)
    cols_to_keep = nonzero_ratio[nonzero_ratio >= min_nonzero_ratio].index
    features = features[cols_to_keep]

    # LightGBM用に特徴量名をクリーンアップ（特殊文字を除去）
    def clean_feature_name(name: str) -> str:
        # 英数字とアンダースコア以外を除去
        # return re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
        return re.sub(r":", "_", str(name))

    features.columns = [clean_feature_name(col) for col in features.columns]

    return features


def parse_args():
    parser = argparse.ArgumentParser(description="FS NMF")
    parser.add_argument("--method", type=str, default="nmf")
    parser.add_argument("--n_components", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()

    # Trainデータの読み込みと処理
    filename_train = DATADIR / cfg["response_tbl_train"]
    response_train = pd.read_pickle(filename_train)

    bs_data_train = load_bs_data(docid_list=response_train.index.tolist())
    pl_data_train = load_pl_data(docid_list=response_train.index.tolist())

    features_train = create_feature_vector(
        bs_data_train,
        pl_data_train,
        pivot_columns="label_jp_long_filled",
    )
    print(f"features_train shape: {features_train.shape}")

    n_components = args.n_components

    # モデルの訓練
    seed = 42
    if args.method == "nmf":
        features_train_minus = features_train.copy() * -1
        features_train_minus.columns = features_train_minus.columns + "_down"
        features_train_minus = features_train_minus[features_train_minus > 0].copy()
        features_train_plus = features_train.copy()
        features_train_plus.columns = features_train_plus.columns + "_up"
        features_train_plus = features_train_plus[features_train_plus > 0].copy()

        features_train_all = pd.concat(
            [features_train_minus, features_train_plus],
            axis=1,
        ).fillna(0)

        model = NMF(
            n_components=n_components,
            init="nndsvda",
            random_state=seed,
            # alpha=0,
            l1_ratio=0,
        )
    elif args.method == "pca":
        features_train_all = features_train
        model = PCA(n_components=n_components)
    elif args.method == "autoencoder":
        features_train_all = features_train
        model = AutoencoderWrapper(
            n_components=n_components,
            epochs=100,
            batch_size=256,
            learning_rate=1e-3,
            random_state=seed,
        )
    else:
        raise ValueError(f"Invalid method: {args.method}")

    # Trainデータで変換
    fsdata_dim_reduced_train = pd.DataFrame(
        model.fit_transform(features_train_all),
        index=features_train_all.index,
        columns=[f"{args.method}_{i}" for i in range(n_components)],
    )
    comp = pd.DataFrame(
        model.components_.transpose(),
        index=features_train_all.columns,
    )

    # 全てのdocidの読み込みと処理
    print("Loading all docids...")
    response_all = get_all_response_tbl()
    all_docids = response_all.index.tolist()
    print(f"Total docids: {len(all_docids)}")

    bs_data_all = load_bs_data(docid_list=all_docids)
    pl_data_all = load_pl_data(docid_list=all_docids)

    features_all = create_feature_vector(
        bs_data_all,
        pl_data_all,
        pivot_columns="label_jp_long_filled",
    )
    print(f"features_all shape: {features_all.shape}")

    # 全データを同じ形式に変換
    if args.method == "nmf":
        features_all_minus = features_all.copy() * -1
        features_all_minus.columns = features_all_minus.columns + "_down"
        features_all_minus = features_all_minus[features_all_minus > 0].copy()
        features_all_plus = features_all.copy()
        features_all_plus.columns = features_all_plus.columns + "_up"
        features_all_plus = features_all_plus[features_all_plus > 0].copy()

        features_all_transformed = pd.concat(
            [features_all_minus, features_all_plus],
            axis=1,
        ).fillna(0)

        # Trainデータと同じカラムに揃える
        missing_cols = set(features_train_all.columns) - set(
            features_all_transformed.columns,
        )
        for col in missing_cols:
            features_all_transformed[col] = 0
        features_all_transformed = features_all_transformed[features_train_all.columns]
    elif args.method in ("pca", "autoencoder"):
        features_all_transformed = features_all
        # Trainデータと同じカラムに揃える
        missing_cols = set(features_train_all.columns) - set(
            features_all_transformed.columns,
        )
        for col in missing_cols:
            features_all_transformed[col] = 0
        features_all_transformed = features_all_transformed[features_train_all.columns]

    # 訂正後データ（fraud + restatement）の読み込みと処理
    print("Loading fraud restatement docids...")

    # fraud系 AMD docid
    fraud_docids_amendment, _, _ = get_task_div_pred_fraud_new(
        period_end_dt_start="2020-04-01",
        period_end_dt_end="2025-03-31",
    )
    fraud_docids_amendment = list(dict.fromkeys(fraud_docids_amendment))
    print(f"  Fraud AMD docids: {len(fraud_docids_amendment)}")

    # restatement系 AMD docid（def_train_eval_split_0222.py と同様に全訂正文書を対象）
    restatement_tbl_path = (
        XBRL_PROJDIR
        / "data/3_processed/dataset_2507/restatement/response_tbl_teisei_2507_v260131_with_year.pkl"
    )
    response_tbl_rst = pd.read_pickle(restatement_tbl_path)
    response_tbl_rst = response_tbl_rst.query(
        "year != 'not_found' and year != 'no_pre_file'",
    )
    restatement_docids_amendment = response_tbl_rst.index.tolist()
    print(f"  Restatement AMD docids: {len(restatement_docids_amendment)}")

    # 両方を結合（重複除去・順序保持）
    all_amd_docids = list(
        dict.fromkeys(fraud_docids_amendment + restatement_docids_amendment),
    )
    print(f"  Combined AMD docids (deduplicated): {len(all_amd_docids)}")

    bs_data_fraud = load_bs_data_amd(docid_list=all_amd_docids)
    pl_data_fraud = load_pl_data_amd(docid_list=all_amd_docids)

    features_fraud = create_feature_vector(
        bs_data_fraud,
        pl_data_fraud,
        pivot_columns="label_jp_long_filled",
    )
    print(f"features_fraud shape: {features_fraud.shape}")

    # 訂正後データを同じ形式に変換
    if args.method == "nmf":
        features_fraud_minus = features_fraud.copy() * -1
        features_fraud_minus.columns = features_fraud_minus.columns + "_down"
        features_fraud_minus = features_fraud_minus[features_fraud_minus > 0].copy()
        features_fraud_plus = features_fraud.copy()
        features_fraud_plus.columns = features_fraud_plus.columns + "_up"
        features_fraud_plus = features_fraud_plus[features_fraud_plus > 0].copy()

        features_fraud_transformed = pd.concat(
            [features_fraud_minus, features_fraud_plus],
            axis=1,
        ).fillna(0)

        # Trainデータと同じカラムに揃える
        missing_cols = set(features_train_all.columns) - set(
            features_fraud_transformed.columns,
        )
        for col in missing_cols:
            features_fraud_transformed[col] = 0
        features_fraud_transformed = features_fraud_transformed[
            features_train_all.columns
        ]
    elif args.method in ("pca", "autoencoder"):
        features_fraud_transformed = features_fraud
        # Trainデータと同じカラムに揃える
        missing_cols = set(features_train_all.columns) - set(
            features_fraud_transformed.columns,
        )
        for col in missing_cols:
            features_fraud_transformed[col] = 0
        features_fraud_transformed = features_fraud_transformed[
            features_train_all.columns
        ]

    # 訂正後データで変換（学習済みモデルを使用）
    print("Transforming fraud restatement data...")
    fsdata_dim_reduced_fraud = pd.DataFrame(
        model.transform(features_fraud_transformed),
        index=features_fraud_transformed.index,
        columns=[f"{args.method}_{i}" for i in range(n_components)],
    )
    # 全データで変換（学習済みモデルを使用）
    print("Transforming all data...")
    fsdata_dim_reduced_all = pd.DataFrame(
        model.transform(features_all_transformed),
        index=features_all_transformed.index,
        columns=[f"{args.method}_{i}" for i in range(n_components)],
    )

    # パラメータベースのサブディレクトリを作成
    subdir_name = f"fs_{args.method}_comp{n_components}"
    output_dir = INTERMEDIATEDIR / "feature" / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Trainデータの保存
    fsdata_dim_reduced_train.to_pickle(
        output_dir / f"fsdata_dim_reduced_{args.method}_train.pkl",
    )
    comp.to_pickle(
        output_dir / f"comp_{args.method}_train.pkl",
    )

    # 全データの保存
    fsdata_dim_reduced_all.to_pickle(
        output_dir / f"fsdata_dim_reduced_{args.method}_all.pkl",
    )

    # 訂正後データの保存
    fsdata_dim_reduced_fraud.to_pickle(
        output_dir / f"fsdata_dim_reduced_{args.method}_fraud.pkl",
    )

    print(f"Saved features to: {output_dir}")
    print(f"  Train features shape: {fsdata_dim_reduced_train.shape}")
    print(f"  All features shape: {fsdata_dim_reduced_all.shape}")
    print(f"  Fraud features shape: {fsdata_dim_reduced_fraud.shape}")
    print(f"  Components shape: {comp.shape}")

    plot_feature_comp(comp, n_components)


# %%
if __name__ == "__main__":
    main()

# %%
# %%
