# %%
import argparse
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.append(r"/Users/noro/Documents/Projects/XBRL_common_space_projection")
sys.path.append(r"/Users/noro/Documents/Projects/t_interpretable_fs")

warnings.filterwarnings("ignore")

from src.data import preproc_rst_loader
from src.libs.downstream_task import (
    get_task_div_pred_fraud_new,
    get_task_div_pred_kpi,
    get_task_div_pred_restatement,
    get_task_div_pred_sector,
)
from src.libs.load_dataset import get_all_response_tbl

importlib.reload(preproc_rst_loader)

XBRL_PROJPATH = r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
XBRL_PROJDIR = Path(XBRL_PROJPATH)
DATADIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/data/1_raw")
INTERMEDIATEDIR = Path(
    "/Users/noro/Documents/Projects/t_interpretable_fs/data/2_intermediate",
)
CFGDIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/src")
cfg = yaml.load(open(CFGDIR / "cfg_exp_main.yaml"), Loader=yaml.FullLoader)

# %%


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate distance matrix for dimensionality reduction features",
    )
    parser.add_argument(
        "--feature_subdir",
        type=str,
        required=True,
        help="Feature subdirectory name (e.g., fs_nmf_comp256, text_lda_topics256_maxf500)",
    )
    parser.add_argument(
        "--amd_filename",
        type=str,
        required=True,
        help="AMD/Fraud feature filename (e.g., fsdata_dim_reduced_nmf_fraud.pkl, text_lda_amd.pkl)",
    )
    parser.add_argument(
        "--eval_filename",
        type=str,
        required=True,
        help="All data feature filename (e.g., fsdata_dim_reduced_nmf_all.pkl, text_lda_test.pkl). Use 'all' file to include all docids.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help="Output subdirectory name (defaults to feature_subdir if not specified)",
    )
    parser.add_argument(
        "--ignore_missing_docids",
        action="store_true",
        default=False,
        help=(
            "If set, docids missing from the feature DataFrame are silently dropped "
            "instead of raising a KeyError. Useful when text-source features only "
            "partially cover the docid universe."
        ),
    )
    return parser.parse_args()


def load_dim_reduced_features(
    feature_subdir: str,
    eval_filename: str,
    amd_filename: str,
) -> pd.DataFrame:
    """次元削減済みの特徴量を読み込む

    Args:
        feature_subdir: feature以下のサブディレクトリ名
        eval_filename: eval/all特徴量のファイル名
        amd_filename: 訂正後（fraud）特徴量のファイル名

    Returns:
        全データ + 訂正後データを結合したDataFrame

    """
    feature_dir = INTERMEDIATEDIR / "feature" / feature_subdir

    # 全データと訂正後データの特徴量を読み込み
    eval_features = pd.read_pickle(feature_dir / eval_filename)
    amd_features = pd.read_pickle(feature_dir / amd_filename)

    # 重複チェック
    overlap_docids = set(eval_features.index) & set(amd_features.index)
    print(f"Overlap between eval and amd: {len(overlap_docids)}")

    ## 重複がある場合は、amdを優先して結合（訂正後データを優先）
    # if len(overlap_docids) > 0:
    #    print(f"Removing {len(overlap_docids)} overlapping docids from eval data")
    #    eval_features = eval_features[~eval_features.index.isin(overlap_docids)]

    # カラム不一致の場合は共通カラムのみを使用（NaN混入を防止）
    common_columns = eval_features.columns.intersection(amd_features.columns)
    if len(common_columns) < len(eval_features.columns) or len(common_columns) < len(
        amd_features.columns,
    ):
        print(
            f"  Warning: Column mismatch detected. Using {len(common_columns)} common columns.",
        )
        print(f"    Eval columns: {len(eval_features.columns)}")
        print(f"    AMD columns: {len(amd_features.columns)}")
        eval_features = eval_features[common_columns]
        amd_features = amd_features[common_columns]

    # 結合
    all_features = pd.concat([eval_features, amd_features], axis=0)
    print(f"Loaded features from: {feature_dir}")
    print(f"  Eval/All: {eval_filename} ({eval_features.shape})")
    print(f"  AMD/Fraud: {amd_filename} ({amd_features.shape})")
    print(f"  Combined shape: {all_features.shape}")

    return all_features


def calc_distance_from_retrieval_pool(
    company_features_df: pd.DataFrame,
    docid_tar: list,
    retrieval_pool_docid: list,
    ignore_missing: bool = False,
) -> pd.DataFrame:
    """検索プールからの距離を計算する

    Args:
        company_features_df: 全体の特徴量DataFrame
        docid_tar: ターゲットとなるdocidのリスト
        retrieval_pool_docid: 検索プールのdocidのリスト
        ignore_missing: Trueの場合、特徴量DataFrameに存在しないdocidを
            KeyErrorを発生させずにスキップする（デフォルト: False）

    Returns:
        距離行列をlong形式にしたDataFrame

    """
    feature_index = set(company_features_df.index)

    # 重複を確認
    docid_tar_set = set(docid_tar)
    retrieval_pool_set = set(retrieval_pool_docid)

    if ignore_missing:
        missing_tar = docid_tar_set - feature_index
        missing_pool = retrieval_pool_set - feature_index
        if missing_tar:
            print(
                f"Warning: {len(missing_tar)} target docids not in features, dropping: "
                f"{sorted(missing_tar)[:10]}{'...' if len(missing_tar) > 10 else ''}",
            )
            docid_tar_set -= missing_tar
        if missing_pool:
            print(
                f"Warning: {len(missing_pool)} pool docids not in features, dropping: "
                f"{sorted(missing_pool)[:10]}{'...' if len(missing_pool) > 10 else ''}",
            )
            retrieval_pool_set -= missing_pool

    overlap_docids = docid_tar_set & retrieval_pool_set
    print(f"Overlap between target and pool: {len(overlap_docids)}")

    if len(overlap_docids) > 0:
        print(f"Warning: {len(overlap_docids)} docids are in both target and pool")

    df1 = company_features_df.loc[list(docid_tar_set)]
    df2 = company_features_df.loc[list(retrieval_pool_set) + list(docid_tar_set)]
    print(f"Target shape: {df1.shape}")
    print(f"Pool shape: {df2.shape}")

    # 数値型のカラムのみを選択
    df1 = df1.select_dtypes(include=[np.number])
    df2 = df2.select_dtypes(include=[np.number])
    print(f"After numeric selection - Target: {df1.shape}, Pool: {df2.shape}")

    # NaN/Infチェック（特徴量が壊れている場合の早期検出）
    nan_count_df1 = np.sum(~np.isfinite(df1.values))
    nan_count_df2 = np.sum(~np.isfinite(df2.values))
    if nan_count_df1 > 0 or nan_count_df2 > 0:
        print(
            f"WARNING: NaN/Inf detected in features! "
            f"Target: {nan_count_df1}, Pool: {nan_count_df2}. "
            f"Distance calculation will produce NaN. "
            f"Please regenerate feature files (model may have collapsed)."
        )

    # 相関係数を距離として計算
    correlation_matrix = np.corrcoef(df1.values, df2.values)[: len(df1), len(df1) :]
    distance_df = pd.DataFrame(
        correlation_matrix,
        index=df1.index.tolist(),
        columns=df2.index.tolist(),
    )
    distance_df.index.name = "docid_source"
    distance_df.columns.name = "docid_target"
    # unstackではなくstack + reset_indexで変換（重複対応）
    distance_df = distance_df.stack().to_frame(name="dist").reset_index()

    if len(distance_df) == 0:
        print(
            "WARNING: Distance DataFrame is empty after stack (likely all NaN from corrcoef). "
            "Feature file may be corrupted. Please re-run feature generation."
        )
    return distance_df


def get_task_div_pred_fraud(period_end_dt_start: str, period_end_dt_end: str):
    """Fraud検出タスクのデータ分割を取得

    Returns:
        docid_fraud_amd: 訂正後のdocidリスト
        docid_fraud: 訂正前のdocidリスト

    """
    response_tbl_all = get_all_response_tbl().query(
        "period_end_dt >= @period_end_dt_start and period_end_dt <= @period_end_dt_end",
    )
    print(period_end_dt_start, " pool size: ", len(response_tbl_all))
    assert len(response_tbl_all) > 0, f"no pool data for {period_end_dt_start}"

    filename = (
        XBRL_PROJDIR
        / "data/3_processed/dataset_2507/restatement/preproc_log_with_amd_docid_fraud_correct.pkl"
    )
    amd_docid_fraud = pd.read_pickle(filename)
    amd_docid_fraud = amd_docid_fraud.query(
        "amendment_document in @response_tbl_all.index",
    )  # 重要でないものを除外
    print(period_end_dt_start, " fraud size: ", len(amd_docid_fraud))
    # assert len(amd_docid_fraud) > 0, f"no fraud data for {itr_year}"
    return amd_docid_fraud.docid.to_list(), amd_docid_fraud.amendment_document.to_list()


def main_all():
    train_periodend_startdate = cfg["train_periodend_startdate"]
    train_periodend_enddate = cfg["train_periodend_enddate"]
    eval_periodend_startdate = cfg["eval_periodend_startdate"]
    eval_periodend_enddate = cfg["eval_periodend_enddate"]
    args = parse_args()

    # 次元削減済みの特徴量を読み込み
    company_features_df = load_dim_reduced_features(
        args.feature_subdir,
        args.eval_filename,
        args.amd_filename,
    )

    # 出力先ディレクトリの作成
    output_dir_name = args.output_subdir or args.feature_subdir
    output_dir = INTERMEDIATEDIR / "distance" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # ===== Sector prediction task =====
    print("\n===== Processing Sector Prediction Task =====")
    (
        eval_sector_test,
        retrieval_pool_sector,
    ) = get_task_div_pred_sector()

    # DataFrameからdocidリストを取得
    eval_sector_test_docid = eval_sector_test.index.tolist()
    retrieval_pool_sector_docid = retrieval_pool_sector.index.tolist()

    print(f"Eval sector test docid count: {len(eval_sector_test_docid)}")
    print(f"Retrieval pool sector docid count: {len(retrieval_pool_sector_docid)}")
    print(
        f"Overlap with features (eval): {len(set(company_features_df.index) & set(eval_sector_test_docid))}",
    )
    print(
        f"Overlap with features (pool): {len(set(company_features_df.index) & set(retrieval_pool_sector_docid))}",
    )

    # evalとpoolの重複を確認
    overlap_docids = set(eval_sector_test_docid) & set(retrieval_pool_sector_docid)
    print(f"Overlap between eval and pool: {len(overlap_docids)}")

    # 重複がある場合は警告を表示（重複はそのまま残す）
    if len(overlap_docids) > 0:
        print(f"Warning: {len(overlap_docids)} docids are in both eval and pool")

    df1 = company_features_df.loc[eval_sector_test_docid]
    df2 = company_features_df.loc[retrieval_pool_sector_docid]

    # 数値型のカラムのみを選択
    df1 = df1.select_dtypes(include=[np.number])
    df2 = df2.select_dtypes(include=[np.number])

    # 相関係数を計算
    correlation_matrix = np.corrcoef(df1.values, df2.values)[: len(df1), len(df1) :]

    distance_df = pd.DataFrame(
        correlation_matrix,
        index=df1.index.tolist(),
        columns=df2.index.tolist(),
    )
    distance_df.index.name = "docid_source"
    distance_df.columns.name = "docid_target"

    # unstackではなくstack + reset_indexで変換（重複対応）
    distance_df = distance_df.stack().to_frame(name="dist").reset_index()
    distance_df.to_pickle(output_dir / "distance_sector.pkl")
    print(f"Saved: {output_dir / 'distance_sector.pkl'}")

    # ===== KPI prediction task =====
    print("\n===== Processing KPI Prediction Task =====")
    (
        eval_response_df,
        retrieval_pool_df,
        eval_response_5_df,
        retrieval_pool_response_5_df,
    ) = get_task_div_pred_kpi(
        eval_periodend_startdate,
        eval_periodend_enddate,
        train_periodend_startdate,
        train_periodend_enddate,
    )

    print(f"Eval response docid count: {len(eval_response_df.index)}")
    print(f"Retrieval pool docid count: {len(retrieval_pool_df.index)}")

    assert len(set(company_features_df.index) & set(eval_response_df.index)) > 0
    assert len(set(company_features_df.index) & set(retrieval_pool_df.index)) > 0

    # evalとpoolの重複を確認
    eval_docids = eval_response_df.index.tolist()
    pool_docids = retrieval_pool_df.index.tolist()
    overlap_docids = set(eval_docids) & set(pool_docids)
    print(f"Overlap between eval and pool: {len(overlap_docids)}")

    if len(overlap_docids) > 0:
        print(f"Warning: {len(overlap_docids)} docids are in both eval and pool")

    df1 = company_features_df.loc[eval_response_df.index]
    df2 = company_features_df.loc[
        retrieval_pool_df.index.tolist() + eval_response_df.index.tolist()
    ]
    print(f"Target shape: {df1.shape}")
    print(f"Pool shape: {df2.shape}")

    # 数値型のカラムのみを選択
    df1 = df1.select_dtypes(include=[np.number])
    df2 = df2.select_dtypes(include=[np.number])
    print(f"After numeric selection - Target: {df1.shape}, Pool: {df2.shape}")

    # 相関係数を計算
    correlation_matrix = np.corrcoef(df1.values, df2.values)[: len(df1), len(df1) :]
    distance_df = pd.DataFrame(
        correlation_matrix,
        index=df1.index.tolist(),
        columns=df2.index.tolist(),
    )
    distance_df.index.name = "docid_source"
    distance_df.columns.name = "docid_target"
    # unstackではなくstack + reset_indexで変換（重複対応）
    distance_df = distance_df.stack().to_frame(name="dist").reset_index()
    distance_df.to_pickle(output_dir / "distance_kpi.pkl")
    print(f"Saved: {output_dir / 'distance_kpi.pkl'}")

    # ===== Fraud detection task =====
    print("\n===== Processing Fraud Detection Task =====")
    for period_end_dt_start, period_end_dt_end in [
        ("2020-04-01", "2021-03-31"),
        ("2021-04-01", "2022-03-31"),
        ("2022-04-01", "2023-03-31"),
        ("2023-04-01", "2024-03-31"),
        ("2024-04-01", "2025-03-31"),
    ]:
        print(
            f"\n--- Processing period: {period_end_dt_start} to {period_end_dt_end} ---",
        )

        # fraud and amendment docid
        docid_fraud_amd, docid_fraud, fraud_docid_all = get_task_div_pred_fraud_new(
            period_end_dt_start,
            period_end_dt_end,
        )
        docid_restatement_amd, docid_restatement = get_task_div_pred_restatement(
            period_end_dt_start,
            period_end_dt_end,
        )
        # retrieval pool docid
        filename = DATADIR / cfg["response_tbl_train"]
        response_tbl_train = pd.read_pickle(filename)
        response_tbl_all = get_all_response_tbl()
        retrieval_pool_docid = response_tbl_all.query(
            "response_edinetCode in @response_tbl_train.response_edinetCode and period_end_dt >= @period_end_dt_start and period_end_dt <= @period_end_dt_end",
        ).index.tolist()

        # oos pool docid
        filename = DATADIR / cfg["response_tbl_test"]
        response_test = pd.read_pickle(filename)
        response_test_pool = list(
            set(response_test.query("task_kpi_flg == 1").response_edinetCode),
        )
        oos_pool_docid = response_tbl_all.query(
            "response_edinetCode in @response_test_pool and period_end_dt >= @period_end_dt_start and period_end_dt <= @period_end_dt_end",
        ).index.tolist()

        # assert with detailed error messages
        overlap_fraud_amd = len(set(company_features_df.index) & set(docid_fraud_amd))
        overlap_pool = len(set(company_features_df.index) & set(retrieval_pool_docid))
        overlap_oos = len(set(company_features_df.index) & set(oos_pool_docid))

        print(
            f"Overlap with features - Fraud AMD: {overlap_fraud_amd}/{len(docid_fraud_amd)}",
        )
        print(
            f"Overlap with features - Pool: {overlap_pool}/{len(retrieval_pool_docid)}",
        )
        print(f"Overlap with features - OOS: {overlap_oos}/{len(oos_pool_docid)}")

        if overlap_fraud_amd == 0:
            print("\nERROR: No fraud/amendment docids found in features!")
            print("This likely means you're using 'eval' file instead of 'all' file.")
            print(
                "Please use --eval_filename with 'all' file (e.g., fsdata_dim_reduced_nmf_all.pkl)",
            )
            print(
                "and --amd_filename with 'fraud' file (e.g., fsdata_dim_reduced_nmf_fraud.pkl)",
            )

        assert overlap_fraud_amd > 0, "No fraud/amendment docids found in features"
        assert overlap_pool > 0, "No retrieval pool docids found in features"
        assert overlap_oos > 0, "No OOS pool docids found in features"

        # fraud_amd distance
        distance_df = calc_distance_from_retrieval_pool(
            company_features_df,
            docid_fraud_amd,
            retrieval_pool_docid,
            ignore_missing=args.ignore_missing_docids,
        )
        distance_df.to_pickle(
            output_dir
            / f"distance_fraud_amd_{period_end_dt_start}_{period_end_dt_end}.pkl",
        )
        print(
            f"Saved: {output_dir / f'distance_fraud_amd_{period_end_dt_start}_{period_end_dt_end}.pkl'}",
        )

        # fraud distance
        distance_df = calc_distance_from_retrieval_pool(
            company_features_df,
            fraud_docid_all,
            retrieval_pool_docid,
            ignore_missing=args.ignore_missing_docids,
        )
        distance_df.to_pickle(
            output_dir
            / f"distance_fraud_{period_end_dt_start}_{period_end_dt_end}.pkl",
        )
        print(
            f"Saved: {output_dir / f'distance_fraud_{period_end_dt_start}_{period_end_dt_end}.pkl'}",
        )

        # oos distance
        distance_df = calc_distance_from_retrieval_pool(
            company_features_df,
            oos_pool_docid,
            retrieval_pool_docid,
            ignore_missing=args.ignore_missing_docids,
        )
        distance_df.to_pickle(
            output_dir / f"distance_oos_{period_end_dt_start}_{period_end_dt_end}.pkl",
        )
        print(
            f"Saved: {output_dir / f'distance_oos_{period_end_dt_start}_{period_end_dt_end}.pkl'}",
        )

        # restatement distance
        distance_df = calc_distance_from_retrieval_pool(
            company_features_df,
            docid_restatement,
            retrieval_pool_docid,
            ignore_missing=args.ignore_missing_docids,
        )
        distance_df.to_pickle(
            output_dir
            / f"distance_restatement_{period_end_dt_start}_{period_end_dt_end}.pkl",
        )
        print(
            f"Saved: {output_dir / f'distance_restatement_{period_end_dt_start}_{period_end_dt_end}.pkl'}",
        )

        # restatement_amd distance
        distance_df = calc_distance_from_retrieval_pool(
            company_features_df,
            docid_restatement_amd,
            retrieval_pool_docid,
            ignore_missing=args.ignore_missing_docids,
        )
        distance_df.to_pickle(
            output_dir
            / f"distance_restatement_amd_{period_end_dt_start}_{period_end_dt_end}.pkl",
        )
        print(
            f"Saved: {output_dir / f'distance_restatement_amd_{period_end_dt_start}_{period_end_dt_end}.pkl'}",
        )

    print("\n===== All tasks completed! =====")
    print(f"Output directory: {output_dir}")


def main():
    args = parse_args()

    # 次元削減済みの特徴量を読み込み
    company_features_df = load_dim_reduced_features(
        args.feature_subdir,
        args.eval_filename,
        args.amd_filename,
    )

    # 出力先ディレクトリの作成
    output_dir_name = args.output_subdir or args.feature_subdir
    output_dir = INTERMEDIATEDIR / "distance" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # ===== Fraud detection task =====
    print("\n===== Processing Fraud Detection Task =====")
    for period_end_dt_start, period_end_dt_end in [
        ("2020-04-01", "2021-03-31"),
        ("2021-04-01", "2022-03-31"),
        ("2022-04-01", "2023-03-31"),
        ("2023-04-01", "2024-03-31"),
        ("2024-04-01", "2025-03-31"),
    ]:
        print(
            f"\n--- Processing period: {period_end_dt_start} to {period_end_dt_end} ---",
        )

        # fraud and amendment docid
        docid_fraud_amd, docid_fraud, fraud_docid_all = get_task_div_pred_fraud_new(
            period_end_dt_start,
            period_end_dt_end,
        )
        docid_restatement_amd, docid_restatement = get_task_div_pred_restatement(
            period_end_dt_start,
            period_end_dt_end,
        )
        # retrieval pool docid
        filename = DATADIR / cfg["response_tbl_train"]
        response_tbl_train = pd.read_pickle(filename)
        response_tbl_all = get_all_response_tbl()
        retrieval_pool_docid = response_tbl_all.query(
            "response_edinetCode in @response_tbl_train.response_edinetCode and period_end_dt >= @period_end_dt_start and period_end_dt <= @period_end_dt_end",
        ).index.tolist()

        # oos pool docid
        filename = DATADIR / cfg["response_tbl_test"]
        response_test = pd.read_pickle(filename)
        response_test_pool = list(
            set(response_test.query("task_kpi_flg == 1").response_edinetCode),
        )
        oos_pool_docid = response_tbl_all.query(
            "response_edinetCode in @response_test_pool and period_end_dt >= @period_end_dt_start and period_end_dt <= @period_end_dt_end",
        ).index.tolist()

        # assert with detailed error messages
        overlap_fraud_amd = len(set(company_features_df.index) & set(docid_fraud_amd))
        overlap_pool = len(set(company_features_df.index) & set(retrieval_pool_docid))
        overlap_oos = len(set(company_features_df.index) & set(oos_pool_docid))

        print(
            f"Overlap with features - Fraud AMD: {overlap_fraud_amd}/{len(docid_fraud_amd)}",
        )
        print(
            f"Overlap with features - Pool: {overlap_pool}/{len(retrieval_pool_docid)}",
        )
        print(f"Overlap with features - OOS: {overlap_oos}/{len(oos_pool_docid)}")

        if overlap_fraud_amd == 0:
            print("\nERROR: No fraud/amendment docids found in features!")
            print("This likely means you're using 'eval' file instead of 'all' file.")
            print(
                "Please use --eval_filename with 'all' file (e.g., fsdata_dim_reduced_nmf_all.pkl)",
            )
            print(
                "and --amd_filename with 'fraud' file (e.g., fsdata_dim_reduced_nmf_fraud.pkl)",
            )

        assert overlap_fraud_amd > 0, "No fraud/amendment docids found in features"
        assert overlap_pool > 0, "No retrieval pool docids found in features"
        assert overlap_oos > 0, "No OOS pool docids found in features"

        # fraud_amd distance
        distance_df = calc_distance_from_retrieval_pool(
            company_features_df,
            docid_fraud_amd,
            retrieval_pool_docid,
            ignore_missing=args.ignore_missing_docids,
        )
        distance_df.to_pickle(
            output_dir
            / f"distance_fraud_amd_{period_end_dt_start}_{period_end_dt_end}.pkl",
        )
        print(
            f"Saved: {output_dir / f'distance_fraud_amd_{period_end_dt_start}_{period_end_dt_end}.pkl'}",
        )

        # fraud distance
        distance_df = calc_distance_from_retrieval_pool(
            company_features_df,
            fraud_docid_all,
            retrieval_pool_docid,
            ignore_missing=args.ignore_missing_docids,
        )
        distance_df.to_pickle(
            output_dir
            / f"distance_fraud_{period_end_dt_start}_{period_end_dt_end}.pkl",
        )
        print(
            f"Saved: {output_dir / f'distance_fraud_{period_end_dt_start}_{period_end_dt_end}.pkl'}",
        )

        # oos distance
        distance_df = calc_distance_from_retrieval_pool(
            company_features_df,
            oos_pool_docid,
            retrieval_pool_docid,
            ignore_missing=args.ignore_missing_docids,
        )
        distance_df.to_pickle(
            output_dir / f"distance_oos_{period_end_dt_start}_{period_end_dt_end}.pkl",
        )
        print(
            f"Saved: {output_dir / f'distance_oos_{period_end_dt_start}_{period_end_dt_end}.pkl'}",
        )

        # restatement distance
        distance_df = calc_distance_from_retrieval_pool(
            company_features_df,
            docid_restatement,
            retrieval_pool_docid,
            ignore_missing=args.ignore_missing_docids,
        )
        distance_df.to_pickle(
            output_dir
            / f"distance_restatement_{period_end_dt_start}_{period_end_dt_end}.pkl",
        )
        print(
            f"Saved: {output_dir / f'distance_restatement_{period_end_dt_start}_{period_end_dt_end}.pkl'}",
        )

        # restatement_amd distance
        distance_df = calc_distance_from_retrieval_pool(
            company_features_df,
            docid_restatement_amd,
            retrieval_pool_docid,
            ignore_missing=args.ignore_missing_docids,
        )
        distance_df.to_pickle(
            output_dir
            / f"distance_restatement_amd_{period_end_dt_start}_{period_end_dt_end}.pkl",
        )
        print(
            f"Saved: {output_dir / f'distance_restatement_amd_{period_end_dt_start}_{period_end_dt_end}.pkl'}",
        )

    print("\n===== All tasks completed! =====")
    print(f"Output directory: {output_dir}")


# %%
if __name__ == "__main__":
    main_all()

# %%
