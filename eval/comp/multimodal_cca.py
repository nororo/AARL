# %%
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

DATADIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/data/1_raw")
INTERMEDIATEDIR = Path(
    "/Users/noro/Documents/Projects/t_interpretable_fs/data/2_intermediate",
)


# %%
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
    # サブディレクトリを使用
    subdir = feature_dir / f"fs_{method}_comp{n_components}"
    filename = subdir / f"fsdata_dim_reduced_{method}_{split}.pkl"
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    fs_features = pd.read_pickle(filename)

    # コンポーネントはtrain時のみ読み込む
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
    # サブディレクトリを試す（LDAの場合）
    if method == "lda":
        subdir = feature_dir / f"text_lda_topics{n_topics}_maxf{max_features}"
        filename = subdir / f"text_{method}_{split}.pkl"
        if not filename.exists():
            # フォールバック：直接ファイルを探す
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

    # コンポーネントを読み込む
    text_comp = None
    comp_file = subdir / f"text_{method}_comp.pkl" if subdir.exists() else None
    if comp_file is None or not comp_file.exists():
        comp_file = feature_dir / f"text_{method}_comp.pkl"
    if comp_file.exists():
        text_comp = pd.read_pickle(comp_file)

    print(f"Loaded Text features ({method}, {split}): {text_features.shape}")
    return text_features, text_comp


# %%
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
    # 共通のdocidを取得
    common_docids = fs_features.index.intersection(text_features.index)
    print(f"Common docids: {len(common_docids)}")
    print(f"  FS only: {len(fs_features.index) - len(common_docids)}")
    print(f"  Text only: {len(text_features.index) - len(common_docids)}")

    # 共通のdocidでフィルタし、同じ順序に揃える
    fs_aligned = fs_features.loc[common_docids].sort_index()
    text_aligned = text_features.loc[common_docids].sort_index()

    return fs_aligned, text_aligned


# %%
def apply_cca(
    fs_train: pd.DataFrame,
    text_train: pd.DataFrame,
    fs_test: pd.DataFrame | None = None,
    text_test: pd.DataFrame | None = None,
    n_components: int = 32,
    standardize: bool = True,
    use_ica: bool = False,
    ica_n_components: int | None = None,
    ica_max_iter: int = 200,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    CCA,
    StandardScaler,
    StandardScaler,
    FastICA | None,
    FastICA | None,
]:
    """CCAを適用して共通空間に投影する

    Args:
        fs_train: 訓練用財務諸表特徴量
        text_train: 訓練用テキスト特徴量
        fs_test: テスト用財務諸表特徴量
        text_test: テスト用テキスト特徴量
        n_components: CCAの成分数
        standardize: 標準化するかどうか
        use_ica: CCA後にICAを適用するかどうか
        ica_n_components: ICAの成分数（Noneの場合はCCAと同じ）
        ica_max_iter: ICAの最大反復回数

    Returns:
        訓練用結合特徴量、テスト用結合特徴量、CCAモデル、FS用スケーラー、Text用スケーラー、
        FS用ICAモデル、Text用ICAモデル

    """
    # 標準化
    fs_scaler = StandardScaler() if standardize else None
    text_scaler = StandardScaler() if standardize else None

    if standardize:
        print("\nStandardizing features...")
        fs_train_scaled = fs_scaler.fit_transform(fs_train)
        text_train_scaled = text_scaler.fit_transform(text_train)
    else:
        fs_train_scaled = fs_train.to_numpy()
        text_train_scaled = text_train.to_numpy()

    # CCAの適用
    print(f"\nApplying CCA with {n_components} components...")
    cca = CCA(n_components=n_components, max_iter=1000)
    fs_train_cca, text_train_cca = cca.fit_transform(
        fs_train_scaled,
        text_train_scaled,
    )

    print(
        f"CCA correlation score (train): {cca.score(fs_train_scaled, text_train_scaled):.4f}",
    )

    # ICAの適用（オプション）
    fs_ica = None
    text_ica = None
    if use_ica:
        if ica_n_components is None:
            ica_n_components = n_components

        print(f"\nApplying ICA with {ica_n_components} components...")
        print(f"  max_iter: {ica_max_iter}")

        # FS側のICA
        fs_ica = FastICA(
            n_components=ica_n_components,
            max_iter=ica_max_iter,
            random_state=42,
        )
        fs_train_cca = fs_ica.fit_transform(fs_train_cca)
        print("FS ICA fitted")

        # Text側のICA
        text_ica = FastICA(
            n_components=ica_n_components,
            max_iter=ica_max_iter,
            random_state=42,
        )
        text_train_cca = text_ica.fit_transform(text_train_cca)
        print("Text ICA fitted")

    # 訓練データの結合
    combined_train = _combine_projections(
        fs_train_cca,
        text_train_cca,
        fs_train.index,
        ica_n_components if use_ica else n_components,
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

        fs_test_cca, text_test_cca = cca.transform(fs_test_scaled, text_test_scaled)
        print(
            f"CCA correlation score (test): {cca.score(fs_test_scaled, text_test_scaled):.4f}",
        )

        # ICAの適用（テストデータ）
        if use_ica:
            fs_test_cca = fs_ica.transform(fs_test_cca)
            text_test_cca = text_ica.transform(text_test_cca)

        combined_test = _combine_projections(
            fs_test_cca,
            text_test_cca,
            fs_test.index,
            ica_n_components if use_ica else n_components,
            "test",
        )

    return combined_train, combined_test, cca, fs_scaler, text_scaler, fs_ica, text_ica


def _combine_projections(
    fs_cca: pd.DataFrame,
    text_cca: pd.DataFrame,
    docids: pd.Index,
    n_components: int,
    split_name: str,
) -> pd.DataFrame:
    """CCA投影を結合してDataFrameを作成する"""
    # 財務諸表のCCA投影
    fs_cca_df = pd.DataFrame(
        fs_cca,
        index=docids,
        columns=[f"fs_cca_{i:02d}" for i in range(n_components)],
    )

    # テキストのCCA投影
    text_cca_df = pd.DataFrame(
        text_cca,
        index=docids,
        columns=[f"text_cca_{i:02d}" for i in range(n_components)],
    )

    # 結合
    combined = pd.concat([fs_cca_df, text_cca_df], axis=1)
    print(f"\nCombined CCA features ({split_name}): {combined.shape}")

    return combined


def transform_with_cca(
    fs_features: pd.DataFrame,
    text_features: pd.DataFrame,
    cca_model: CCA,
    fs_scaler: StandardScaler | None,
    text_scaler: StandardScaler | None,
    fs_ica: FastICA | None = None,
    text_ica: FastICA | None = None,
    split_name: str = "data",
    fs_train_columns: list[str] | None = None,
    text_train_columns: list[str] | None = None,
) -> pd.DataFrame:
    """学習済みCCAモデルを使って新しいデータを変換する

    Args:
        fs_features: 財務諸表特徴量
        text_features: テキスト特徴量
        cca_model: 学習済みCCAモデル
        fs_scaler: FS用スケーラー
        text_scaler: Text用スケーラー
        fs_ica: FS用ICAモデル（オプション）
        text_ica: Text用ICAモデル（オプション）
        split_name: データセット名（ログ用）
        fs_train_columns: 訓練データのFS特徴量カラム（順序を揃えるため）
        text_train_columns: 訓練データのText特徴量カラム（順序を揃えるため）

    Returns:
        CCA変換後の結合特徴量

    """
    # docidを揃える
    fs_aligned, text_aligned = align_features(fs_features, text_features)

    if len(fs_aligned) == 0:
        print(f"Warning: No common docids for {split_name}. Skipping.")
        return None

    # 訓練データとカラムを揃える
    if fs_train_columns is not None:
        missing_cols = set(fs_train_columns) - set(fs_aligned.columns)
        for col in missing_cols:
            fs_aligned[col] = 0
        # 訓練データと同じ順序に揃える（余分なカラムは削除）
        fs_aligned = fs_aligned.reindex(columns=fs_train_columns, fill_value=0)

    if text_train_columns is not None:
        missing_cols = set(text_train_columns) - set(text_aligned.columns)
        for col in missing_cols:
            text_aligned[col] = 0
        # 訓練データと同じ順序に揃える（余分なカラムは削除）
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

    # CCA変換
    fs_cca, text_cca = cca_model.transform(fs_scaled, text_scaled)
    print(
        f"CCA correlation score ({split_name}): {cca_model.score(fs_scaled, text_scaled):.4f}",
    )

    # ICA変換（オプション）
    if fs_ica is not None:
        fs_cca = fs_ica.transform(fs_cca)
    if text_ica is not None:
        text_cca = text_ica.transform(text_cca)

    # 結合
    ica_n_components = fs_cca.shape[1]
    combined = _combine_projections(
        fs_cca,
        text_cca,
        fs_aligned.index,
        ica_n_components,
        split_name,
    )

    return combined


def transform_fs_only_with_cca(
    fs_features: pd.DataFrame,
    cca_model: CCA,
    fs_scaler: StandardScaler | None,
    text_scaler: StandardScaler | None,
    fs_ica: FastICA | None = None,
    split_name: str = "data",
    fs_train_columns: list[str] | None = None,
    text_train_columns: list[str] | None = None,
    n_components: int = 32,
) -> pd.DataFrame:
    """学習済みCCAモデルを使ってFS特徴量のみを変換する（推論用）

    Args:
        fs_features: 財務諸表特徴量
        cca_model: 学習済みCCAモデル
        fs_scaler: FS用スケーラー
        text_scaler: Text用スケーラー
        fs_ica: FS用ICAモデル（オプション）
        split_name: データセット名（ログ用）
        fs_train_columns: 訓練データのFS特徴量カラム
        text_train_columns: 訓練データのText特徴量カラム
        n_components: CCA成分数

    Returns:
        FS特徴量のCCA変換結果のみ

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

    # CCA変換（FS側のみ）
    # x_weightsを使用してFS側のみを変換
    fs_cca = cca_model.x_weights_.T @ fs_scaled.T
    fs_cca = fs_cca.T

    # ICA変換（オプション）
    if fs_ica is not None:
        fs_cca = fs_ica.transform(fs_cca)

    # FS側のみのDataFrameを作成
    fs_cca_df = pd.DataFrame(
        fs_cca,
        index=fs_aligned.index,
        columns=[f"fs_cca_{i:02d}" for i in range(fs_cca.shape[1])],
    )

    print(f"\nFS-only CCA features ({split_name}): {fs_cca_df.shape}")

    return fs_cca_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal CCA: FS features + Text features",
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
        help="Number of CCA components",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        default=True,
        help="Standardize features before CCA",
    )
    parser.add_argument(
        "--use_ica",
        action="store_true",
        default=False,
        help="Apply ICA after CCA for better interpretability",
    )
    parser.add_argument(
        "--ica_n_components",
        type=int,
        default=None,
        help="Number of ICA components (default: same as CCA)",
    )
    parser.add_argument(
        "--ica_max_iter",
        type=int,
        default=200,
        help="ICA maximum iterations",
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
    return parser.parse_args()


def make_feature_multimodal_cca(
    fs_method: str = "nmf",
    text_method: str = "lda",
    n_components: int = 32,
    standardize: bool = True,
    use_ica: bool = False,
    ica_n_components: int | None = None,
    ica_max_iter: int = 200,
    output_suffix: str = "",
    fs_n_components: int = 256,
    text_n_topics: int = 256,
    text_max_features: int = 500,
    process_all_data: bool = True,
    process_fraud_data: bool = True,
):
    """メイン処理：マルチモーダルCCA特徴量を作成して保存

    Args:
        fs_method: 財務諸表の次元削減手法
        text_method: テキスト特徴量抽出手法
        n_components: CCA成分数
        standardize: 標準化するか
        use_ica: ICAを使用するか
        ica_n_components: ICAの成分数
        ica_max_iter: ICAの最大反復回数
        output_suffix: 出力ファイル名サフィックス
        fs_n_components: FS特徴量の次元数
        text_n_topics: テキスト特徴量のトピック数
        text_max_features: テキスト特徴量の最大特徴量数
        process_all_data: 全データへの推論を行うか
        process_fraud_data: 訂正後データへの推論を行うか

    """
    print("=" * 50)
    print("Multimodal CCA Feature Extraction")
    print("=" * 50)
    print(f"FS method: {fs_method}")
    print(f"Text method: {text_method}")
    print(f"CCA components: {n_components}")
    print(f"Standardize: {standardize}")
    print(f"Use ICA: {use_ica}")
    if use_ica:
        print(f"  ICA n_components: {ica_n_components or n_components}")
        print(f"  ICA max_iter: {ica_max_iter}")
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

    # 訓練データのカラム順序を保存（推論時に使用）
    fs_train_columns = fs_train_aligned.columns.tolist()
    text_train_columns = text_train_aligned.columns.tolist()

    # テストデータが存在するか確認
    test_file = feature_dir / f"text_{text_method}_test.pkl"
    has_test_data = test_file.exists()

    fs_test_aligned = None
    text_test_aligned = None

    if has_test_data:
        print("\n--- Loading Test Features ---")
        print(
            "Note: Using training FS features for test data (no separate test FS features)",
        )
        text_test, _ = load_text_features(
            feature_dir,
            split="test",
            method=text_method,
            n_topics=text_n_topics,
            max_features=text_max_features,
        )

        # テストデータのdocidを揃える
        print("\n--- Aligning Test Features ---")
        fs_test_aligned, text_test_aligned = align_features(fs_train, text_test)

        # 共通のdocidがない場合はテストデータをスキップ
        if len(fs_test_aligned) == 0 or len(text_test_aligned) == 0:
            print(
                "Warning: No common docids between FS and Text test data. Skipping test data processing.",
            )
            fs_test_aligned = None
            text_test_aligned = None
    else:
        print("\n--- No Test Data Found ---")

    # CCAの適用
    print("\n--- Applying CCA ---")
    (
        combined_train,
        combined_test,
        cca_model,
        fs_scaler,
        text_scaler,
        fs_ica,
        text_ica,
    ) = apply_cca(
        fs_train=fs_train_aligned,
        text_train=text_train_aligned,
        fs_test=fs_test_aligned,
        text_test=text_test_aligned,
        n_components=n_components,
        standardize=standardize,
        use_ica=use_ica,
        ica_n_components=ica_n_components,
        ica_max_iter=ica_max_iter,
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

            combined_all = transform_fs_only_with_cca(
                fs_features=fs_all,
                cca_model=cca_model,
                fs_scaler=fs_scaler,
                text_scaler=text_scaler,
                fs_ica=fs_ica,
                split_name="all",
                fs_train_columns=fs_train_columns,
                text_train_columns=text_train_columns,
                n_components=ica_n_components if use_ica else n_components,
            )
        except FileNotFoundError as e:
            print(f"Warning: Could not load all data features: {e}")
            combined_all = None

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

            n_comp = ica_n_components if use_ica else n_components
            parts = []

            if text_fraud is not None:
                # textとfsの共通docidは両方で推論
                common_docids = fs_fraud.index.intersection(text_fraud.index)
                fs_only_docids = fs_fraud.index.difference(text_fraud.index)
                print(f"Fraud data with text:    {len(common_docids)}")
                print(f"Fraud data without text: {len(fs_only_docids)}")

                if len(common_docids) > 0:
                    combined_both = transform_with_cca(
                        fs_features=fs_fraud.loc[common_docids],
                        text_features=text_fraud.loc[common_docids],
                        cca_model=cca_model,
                        fs_scaler=fs_scaler,
                        text_scaler=text_scaler,
                        fs_ica=fs_ica,
                        text_ica=text_ica,
                        split_name="fraud (with text)",
                        fs_train_columns=fs_train_columns,
                        text_train_columns=text_train_columns,
                    )
                    parts.append(combined_both)

                if len(fs_only_docids) > 0:
                    fs_only_result = transform_fs_only_with_cca(
                        fs_features=fs_fraud.loc[fs_only_docids],
                        cca_model=cca_model,
                        fs_scaler=fs_scaler,
                        text_scaler=text_scaler,
                        fs_ica=fs_ica,
                        split_name="fraud (FS only)",
                        fs_train_columns=fs_train_columns,
                        text_train_columns=text_train_columns,
                        n_components=n_comp,
                    )
                    # text側をゼロ埋めしてshapeを統一
                    for i in range(n_comp):
                        fs_only_result[f"text_cca_{i:02d}"] = 0.0
                    parts.append(fs_only_result)
            else:
                # textデータなし：全件fsのみで推論
                fs_only_result = transform_fs_only_with_cca(
                    fs_features=fs_fraud,
                    cca_model=cca_model,
                    fs_scaler=fs_scaler,
                    text_scaler=text_scaler,
                    fs_ica=fs_ica,
                    split_name="fraud (FS only)",
                    fs_train_columns=fs_train_columns,
                    text_train_columns=text_train_columns,
                    n_components=n_comp,
                )
                # text側をゼロ埋めしてshapeを統一
                for i in range(n_comp):
                    fs_only_result[f"text_cca_{i:02d}"] = 0.0
                parts.append(fs_only_result)

            if parts:
                combined_fraud = pd.concat(parts).sort_index()

        except FileNotFoundError as e:
            print(f"Warning: Could not load fraud data features: {e}")
            combined_fraud = None

    # 保存
    print("\n--- Saving Results ---")

    # パラメータに基づいてサブディレクトリを作成
    cca_type = "cca_ica" if use_ica else "cca"
    dir_parts = [cca_type, fs_method, text_method, f"comp{n_components}"]
    if use_ica:
        ica_comp = ica_n_components or n_components
        dir_parts.append(f"ica{ica_comp}")
    if output_suffix:
        dir_parts.append(output_suffix)

    subdir_name = "_".join(dir_parts)
    output_dir = feature_dir / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    base_name = f"multimodal_{cca_type}_{fs_method}_{text_method}"

    train_output_path = output_dir / f"{base_name}_train.pkl"
    combined_train.to_pickle(train_output_path)
    print(f"Training features: {train_output_path}")

    if combined_test is not None:
        test_output_path = output_dir / f"{base_name}_test.pkl"
        combined_test.to_pickle(test_output_path)
        print(f"Test features: {test_output_path}")

    # 全データの保存
    if combined_all is not None:
        all_output_path = output_dir / f"{base_name}_all.pkl"
        combined_all.to_pickle(all_output_path)
        print(f"All data features: {all_output_path}")

    # 訂正後データの保存
    if combined_fraud is not None:
        fraud_output_path = output_dir / f"{base_name}_fraud.pkl"
        combined_fraud.to_pickle(fraud_output_path)
        print(f"Fraud data features: {fraud_output_path}")

    # モデルとスケーラーの保存
    model_output_path = output_dir / f"{base_name}_model.pkl"
    model_dict = {
        "cca": cca_model,
        "fs_scaler": fs_scaler,
        "text_scaler": text_scaler,
        "n_components": n_components,
        "fs_method": fs_method,
        "text_method": text_method,
        "use_ica": use_ica,
        "fs_ica": fs_ica,
        "text_ica": text_ica,
        "fs_train_columns": fs_train_columns,
        "text_train_columns": text_train_columns,
    }
    if use_ica:
        model_dict.update(
            {
                "ica_n_components": ica_n_components or n_components,
                "ica_max_iter": ica_max_iter,
            },
        )

    # CCA係数の保存
    weights_output_path = output_dir / f"{base_name}_weights.pkl"
    pd.to_pickle(
        {
            "x_weights": pd.DataFrame(
                cca_model.x_weights_,
                index=fs_train_aligned.columns,
                columns=[f"comp_{i:02d}" for i in range(n_components)],
            ),
            "y_weights": pd.DataFrame(
                cca_model.y_weights_,
                index=text_train_aligned.columns,
                columns=[f"comp_{i:02d}" for i in range(n_components)],
            ),
        },
        weights_output_path,
    )
    print(f"CCA weights: {weights_output_path}")

    pd.to_pickle(model_dict, model_output_path)
    print(f"CCA model: {model_output_path}")

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

    # 整列された特徴量の保存（可視化用）
    aligned_output_path = output_dir / f"{base_name}_aligned.pkl"
    pd.to_pickle(
        {
            "fs_train_aligned": fs_train_aligned,
            "text_train_aligned": text_train_aligned,
        },
        aligned_output_path,
    )
    print(f"Aligned features: {aligned_output_path}")

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
    print(f"  FS CCA features: {n_components}")
    print(f"  Text CCA features: {n_components}")

    return {
        "combined_train": combined_train,
        "combined_test": combined_test,
        "combined_all": combined_all,
        "combined_fraud": combined_fraud,
        "cca_model": cca_model,
        "fs_train_aligned": fs_train_aligned,
        "text_train_aligned": text_train_aligned,
        "fs_comp": fs_comp,
        "text_comp": text_comp,
        "fs_ica": fs_ica,
        "text_ica": text_ica,
        "fs_scaler": fs_scaler,
        "text_scaler": text_scaler,
    }


# %%
def main():
    """コマンドライン実行用のメイン関数"""
    args = parse_args()
    results = make_feature_multimodal_cca(
        fs_method=args.fs_method,
        text_method=args.text_method,
        n_components=args.n_components,
        standardize=args.standardize,
        use_ica=args.use_ica,
        ica_n_components=args.ica_n_components,
        ica_max_iter=args.ica_max_iter,
        output_suffix=args.output_suffix,
        fs_n_components=args.fs_n_components,
        text_n_topics=args.text_n_topics,
        text_max_features=args.text_max_features,
        process_all_data=args.process_all_data,
        process_fraud_data=args.process_fraud_data,
    )
    return results


# %%
if __name__ == "__main__":
    results = main()

## %%
## Jupyter実行例
## 標準CCA（全データと訂正後データへの推論を含む）
# results = make_feature_multimodal_cca()
#
## 結果の取り出し例
# combined_train = results["combined_train"]
# combined_test = results["combined_test"]
# combined_all = results["combined_all"]
# combined_fraud = results["combined_fraud"]
# cca_model = results["cca_model"]
#
## %%
## CCA-ICA使用例（全データと訂正後データへの推論を含む）
# results_ica = make_feature_multimodal_cca(
#    use_ica=True,
#    n_components=32,
#    ica_n_components=32,
# )
#
## %%
## 訓練データのみ処理する例（高速）
# results_train_only = make_feature_multimodal_cca(
#    process_all_data=False,
#    process_fraud_data=False,
# )

# %%
