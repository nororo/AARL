# %%
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_auc_score, roc_curve

XBRL_PROJPATH = r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
XBRL_PROJDIR = Path(XBRL_PROJPATH)
PROCDIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/data/3_processed")
INTERMEDIATEDIR = Path(
    "/Users/noro/Documents/Projects/t_interpretable_fs/data/2_intermediate",
)
DATADIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/data/1_raw")
CFGDIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/src")
cfg = yaml.load(open(CFGDIR / "cfg_exp_add.yaml"), Loader=yaml.FullLoader)

from libs.downstream_task import (
    get_task_div_pred_fraud_new,
    get_task_div_pred_restatement,
)
from libs.load_dataset import get_all_response_tbl


def load_distance_fraud2(
    period_end_dt_start: str,
    period_end_dt_end: str,
    method: str = "correlation",
) -> pd.DataFrame:
    filename = (
        INTERMEDIATEDIR
        / "distance"
        / method
        / f"distance_fraud_{period_end_dt_start}_{period_end_dt_end}.pkl"
    )
    cor_distance_fraud = pd.read_pickle(filename)
    return cor_distance_fraud


def load_distance_fraud_amd2(
    period_end_dt_start: str,
    period_end_dt_end: str,
    method: str = "correlation",
) -> pd.DataFrame:
    filename = (
        INTERMEDIATEDIR
        / "distance"
        / method
        / f"distance_fraud_{period_end_dt_start}_{period_end_dt_end}_amd.pkl"
    )
    cor_distance_fraud_amd = pd.read_pickle(filename)
    return cor_distance_fraud_amd


def load_distance_oos2(
    period_end_dt_start: str,
    period_end_dt_end: str,
    method: str = "correlation",
) -> pd.DataFrame:
    filename = (
        INTERMEDIATEDIR
        / "distance"
        / method
        / f"distance_oos_{period_end_dt_start}_{period_end_dt_end}.pkl"
    )
    cor_distance_oos = pd.read_pickle(filename)
    return cor_distance_oos


def load_distance_restatement2(
    period_end_dt_start: str,
    period_end_dt_end: str,
    method: str = "correlation",
) -> pd.DataFrame:
    filename = (
        INTERMEDIATEDIR
        / "distance"
        / method
        / f"distance_restatement_{period_end_dt_start}_{period_end_dt_end}.pkl"
    )
    cor_distance_restatement = pd.read_pickle(filename)
    return cor_distance_restatement


def load_distance_restatement_amd2(
    period_end_dt_start: str,
    period_end_dt_end: str,
    method: str = "correlation",
) -> pd.DataFrame:
    filename = (
        INTERMEDIATEDIR
        / "distance"
        / method
        / f"distance_restatement_{period_end_dt_start}_{period_end_dt_end}_amd.pkl"
    )
    cor_distance_restatement_amd = pd.read_pickle(filename)
    return cor_distance_restatement_amd


def load_distance_fraud(
    period_end_dt_start: str,
    period_end_dt_end: str,
    method: str = "clip",
) -> pd.DataFrame:
    filename = (
        INTERMEDIATEDIR
        / "distance"
        / method
        / f"distance_fraud_{period_end_dt_start}_{period_end_dt_end}.pkl"
    )
    clip_distance_fraud = (
        pd.read_pickle(filename)
        .rename(
            columns={"docid_source": "source_docid", "docid_target": "target_docid"},
        )
        .query("source_docid != target_docid")
    )
    return clip_distance_fraud


def load_distance_fraud_amd(
    period_end_dt_start: str,
    period_end_dt_end: str,
    method: str = "clip",
) -> pd.DataFrame:
    filename = (
        INTERMEDIATEDIR
        / "distance"
        / method
        / f"distance_fraud_amd_{period_end_dt_start}_{period_end_dt_end}.pkl"
    )
    clip_distance_fraud_amd = (
        pd.read_pickle(filename)
        .rename(
            columns={"docid_source": "source_docid", "docid_target": "target_docid"},
        )
        .query("source_docid != target_docid")
    )
    return clip_distance_fraud_amd


def load_distance_oos(
    period_end_dt_start: str,
    period_end_dt_end: str,
    method: str = "clip",
) -> pd.DataFrame:
    filename = (
        INTERMEDIATEDIR
        / "distance"
        / method
        / f"distance_oos_{period_end_dt_start}_{period_end_dt_end}.pkl"
    )
    clip_distance_oos = (
        pd.read_pickle(filename)
        .rename(
            columns={"docid_source": "source_docid", "docid_target": "target_docid"},
        )
        .query("source_docid != target_docid")
    )
    return clip_distance_oos


def load_distance_restatement(
    period_end_dt_start: str,
    period_end_dt_end: str,
    method: str = "clip",
) -> pd.DataFrame:
    filename = (
        INTERMEDIATEDIR
        / "distance"
        / method
        / f"distance_restatement_{period_end_dt_start}_{period_end_dt_end}.pkl"
    )
    clip_distance_restatement = (
        pd.read_pickle(filename)
        .rename(
            columns={"docid_source": "source_docid", "docid_target": "target_docid"},
        )
        .query("source_docid != target_docid")
    )
    return clip_distance_restatement


def load_distance_restatement_amd(
    period_end_dt_start: str,
    period_end_dt_end: str,
    method: str = "clip",
) -> pd.DataFrame:
    filename = (
        INTERMEDIATEDIR
        / "distance"
        / method
        / f"distance_restatement_amd_{period_end_dt_start}_{period_end_dt_end}.pkl"
    )
    clip_distance_restatement_amd = (
        pd.read_pickle(filename)
        .rename(
            columns={"docid_source": "source_docid", "docid_target": "target_docid"},
        )
        .query("source_docid != target_docid")
    )
    return clip_distance_restatement_amd


# %%
def agg_func_kNN_distance(sr, n_topk=5, col_name="ot_distance"):
    sr = sr.sort_values(by=col_name)
    topk_dist = sr.iloc[:n_topk][col_name].mean()
    return topk_dist


# %%


# %%
def eval_model_fraud(
    n_topk: int = 5,
    N = None,  # noqa: N803
    random_state: int = 0,
) -> pd.DataFrame:
    results = []
    rng = np.random.default_rng(random_state)
    # oos pool docid
    response_tbl_all = get_all_response_tbl()
    filename = DATADIR / cfg["response_tbl_test"]
    response_test = pd.read_pickle(filename)
    response_test_pool = list(
        set(response_test.query("task_kpi_flg == 1").response_edinetCode),
    )

    for period_end_dt_start, period_end_dt_end in [
        ("2020-04-01", "2021-03-31"),
        ("2021-04-01", "2022-03-31"),
        ("2022-04-01", "2023-03-31"),
        ("2023-04-01", "2024-03-31"),
        ("2024-04-01", "2025-03-31"),
    ]:
        amd_docid, amd_docid_fraud, fraud_docid_list = get_task_div_pred_fraud_new(
            period_end_dt_start,
            period_end_dt_end,
        )
        print("amd_docid", len(amd_docid))
        print("fraud_docid_list", len(fraud_docid_list))
        oos_pool_docid = response_tbl_all.query(
            "response_edinetCode in @response_test_pool and period_end_dt >= @period_end_dt_start and period_end_dt <= @period_end_dt_end",
        ).index.tolist()
        print("oos_pool_docid", len(oos_pool_docid))

        calc_list = []
        common_fraud_set = set(fraud_docid_list)
        common_amd_set = set(amd_docid)
        common_oos_set = set(oos_pool_docid)
        for method_name in cfg["method_name_list"]:
            print(f"===== method_name: {method_name} =====")
            distance_oos = load_distance_oos(
                period_end_dt_start,
                period_end_dt_end,
                method=method_name,
            )
            distance_oos["dist"] = 1 - distance_oos["dist"]
            print(
                "oos not calculated:",
                set(oos_pool_docid) - set(distance_oos.source_docid),
            )

            distance_fraud = load_distance_fraud(
                period_end_dt_start,
                period_end_dt_end,
                method=method_name,
            )
            print(
                "fraud not calculated:",
                set(fraud_docid_list) - set(distance_fraud.source_docid),
            )

            print("distance_fraud", len(distance_fraud))
            distance_fraud["dist"] = 1 - distance_fraud["dist"]
            distance_fraud_amd = load_distance_fraud_amd(
                period_end_dt_start,
                period_end_dt_end,
                method=method_name,
            )
            print(
                "amd not calculated:",
                set(amd_docid) - set(distance_fraud_amd.source_docid),
            )
            distance_fraud_amd["dist"] = 1 - distance_fraud_amd["dist"]
            calc_list.append(
                {
                    "method_name": method_name,
                    "distance_df_oos": distance_oos,
                    "distance_df_fraud": distance_fraud,
                    "distance_df_fraud_amd": distance_fraud_amd,
                },
            )
            common_fraud_set = set(distance_fraud.source_docid) & common_fraud_set
            common_amd_set = set(distance_fraud_amd.source_docid) & common_amd_set
            common_oos_set = set(distance_oos.source_docid) & common_oos_set

        # correlation (already converted to distance)
        print("===== correlation =====")
        cor_distance_fraud = load_distance_fraud2(
            period_end_dt_start,
            period_end_dt_end,
            method="correlation",
        )
        print(
            "fraud not calculated:",
            set(fraud_docid_list) - set(cor_distance_fraud.source_docid),
        )
        cor_distance_fraud_amd = load_distance_fraud_amd2(
            period_end_dt_start,
            period_end_dt_end,
            method="correlation",
        )
        print(
            "amd not calculated:",
            set(amd_docid) - set(cor_distance_fraud_amd.source_docid),
        )
        cor_distance_oos = load_distance_oos2(
            period_end_dt_start,
            period_end_dt_end,
            method="correlation",
        )
        print(
            "oos not calculated:",
            set(oos_pool_docid) - set(cor_distance_oos.source_docid),
        )
        calc_list.append(
            {
                "method_name": "correlation",
                "distance_df_oos": cor_distance_oos,
                "distance_df_fraud": cor_distance_fraud,
                "distance_df_fraud_amd": cor_distance_fraud_amd,
            },
        )
        common_fraud_set = set(cor_distance_fraud.source_docid) & common_fraud_set
        common_amd_set = set(cor_distance_fraud_amd.source_docid) & common_amd_set
        common_oos_set = set(cor_distance_oos.source_docid) & common_oos_set
        # mahalanobis
        print("===== mahalanobis =====")
        mahalanobis_distance_fraud = load_distance_fraud2(
            period_end_dt_start,
            period_end_dt_end,
            method="mahalanobis",
        )
        print(
            "fraud not calculated:",
            set(fraud_docid_list) - set(mahalanobis_distance_fraud.source_docid),
        )
        # mahalanobis_distance_fraud["dist"] = 1 - mahalanobis_distance_fraud["dist"]
        mahalanobis_distance_fraud_amd = load_distance_fraud_amd2(
            period_end_dt_start,
            period_end_dt_end,
            method="mahalanobis",
        )
        print(
            "amd not calculated:",
            set(amd_docid) - set(mahalanobis_distance_fraud_amd.source_docid),
        )
        #mahalanobis_distance_fraud_amd["dist"] = (
        #    1 - mahalanobis_distance_fraud_amd["dist"]
        #)
        mahalanobis_distance_oos = load_distance_oos2(
            period_end_dt_start,
            period_end_dt_end,
            method="mahalanobis",
        )
        # mahalanobis_distance_oos["dist"] = 1 - mahalanobis_distance_oos["dist"]
        print(
            "oos not calculated:",
            set(oos_pool_docid) - set(mahalanobis_distance_oos.source_docid),
        )
        calc_list.append(
            {
                "method_name": "mahalanobis",
                "distance_df_fraud": mahalanobis_distance_fraud,
                "distance_df_fraud_amd": mahalanobis_distance_fraud_amd,
                "distance_df_oos": mahalanobis_distance_oos,
            },
        )
        common_fraud_set = (
            set(mahalanobis_distance_fraud.source_docid) & common_fraud_set
        )
        common_amd_set = (
            set(mahalanobis_distance_fraud_amd.source_docid) & common_amd_set
        )
        common_oos_set = set(mahalanobis_distance_oos.source_docid) & common_oos_set
        # BM25
        print("=== BM25 === ")
        bm25_distance_fraud = load_distance_fraud(
            period_end_dt_start,
            period_end_dt_end,
            method="bm25",
        )
        print(
            "fraud not calculated:",
            set(fraud_docid_list) - set(bm25_distance_fraud.source_docid),
        )
        # bm25_distance_fraud["dist"] = 1 - bm25_distance_fraud["dist"]
        bm25_distance_fraud_amd = load_distance_fraud_amd(
            period_end_dt_start,
            period_end_dt_end,
            method="bm25",
        )
        print(
            "amd not calculated:",
            set(amd_docid) - set(bm25_distance_fraud_amd.source_docid),
        )
        # bm25_distance_fraud_amd["dist"] = 1 - bm25_distance_fraud_amd["dist"]
        bm25_distance_oos = load_distance_oos(
            period_end_dt_start,
            period_end_dt_end,
            method="bm25",
        )
        # bm25_distance_oos["dist"] = 1 - bm25_distance_oos["dist"]
        print(
            "oos not calculated:",
            set(oos_pool_docid) - set(bm25_distance_oos.source_docid),
        )
        calc_list.append(
            {
                "method_name": "bm25",
                "distance_df_fraud": bm25_distance_fraud,
                "distance_df_fraud_amd": bm25_distance_fraud_amd,
                "distance_df_oos": bm25_distance_oos,
            },
        )
        common_fraud_set = set(bm25_distance_fraud.source_docid) & common_fraud_set
        common_amd_set = set(bm25_distance_fraud_amd.source_docid) & common_amd_set
        common_oos_set = set(bm25_distance_oos.source_docid) & common_oos_set

        print("common_fraud_set", len(common_fraud_set))
        print("common_amd_set", len(common_amd_set))
        print("common_oos_set", len(common_oos_set))
        if N is None:
            eval_oos_set = common_oos_set
        else:
            if N <= 0:
                msg = "N must be positive or None"
                raise ValueError(msg)
            n_oos_sample = int(np.ceil(len(common_fraud_set) * N))
            n_oos_sample = min(n_oos_sample, len(common_oos_set))
            eval_oos_set = set(
                rng.choice(
                    sorted(common_oos_set),
                    size=n_oos_sample,
                    replace=False,
                ),
            )
            print("eval_oos_set", len(eval_oos_set), f"(N={N})")
        for calc in calc_list:
            results.append(
                {
                    "period": f"{period_end_dt_start}_{period_end_dt_end}",
                    "method": calc["method_name"],
                    "task": "fraud",
                    "score_list": calc["distance_df_fraud"]
                    .query("source_docid in @common_fraud_set and target_docid not in @common_fraud_set and target_docid not in @common_amd_set and target_docid not in @common_oos_set")
                    .groupby("source_docid")
                    .apply(agg_func_kNN_distance, n_topk=n_topk, col_name="dist")
                    .to_list(),
                },
            )
            results.append(
                {
                    "period": f"{period_end_dt_start}_{period_end_dt_end}",
                    "method": calc["method_name"],
                    "task": "fraud_amd",
                    "score_list": calc["distance_df_fraud_amd"]
                    .query("source_docid in @common_amd_set and target_docid not in @common_amd_set and target_docid not in @common_fraud_set and target_docid not in @common_oos_set")
                    .groupby("source_docid")
                    .apply(agg_func_kNN_distance, n_topk=n_topk, col_name="dist")
                    .to_list(),
                },
            )
            results.append(
                {
                    "period": f"{period_end_dt_start}_{period_end_dt_end}",
                    "method": calc["method_name"],
                    "task": "oos",
                    "score_list": calc["distance_df_oos"]
                    .query(
                        "source_docid in @eval_oos_set "
                        "and target_docid not in @common_oos_set "
                        "and target_docid not in @common_fraud_set "
                        "and target_docid not in @common_amd_set",
                    )
                    .groupby("source_docid")
                    .apply(agg_func_kNN_distance, n_topk=n_topk, col_name="dist")
                    .to_list(),
                },
            )

    rst_score = pd.DataFrame(results).explode("score_list")
    return rst_score


# %%
# len(rst_score.query("method == 'clip' and task == 'fraud'"))


# %%
def calc_auc(rst_score):
    # OOSとfraudの分類AUCを計算
    auc_results = []
    for period in rst_score["period"].unique():
        for method in rst_score["method"].unique():
            # OOSのスコアを取得（ラベル=0）
            oos_scores = (
                rst_score.query(
                    "period == @period and method == @method and task == 'oos'",
                )
                .score_list.astype(float)
                .tolist()
            )

            # fraudのスコアを取得（ラベル=1）
            fraud_scores = (
                rst_score.query(
                    "period == @period and method == @method and task == 'fraud'",
                )
                .score_list.astype(float)
                .tolist()
            )
            fraud_scores_amd = (
                rst_score.query(
                    "period == @period and method == @method and task == 'fraud_amd'",
                )
                .score_list.astype(float)
                .tolist()
            )

            if len(oos_scores) > 0 and len(fraud_scores) > 0:
                # ラベルとスコアを結合
                y_true = [0] * len(oos_scores) + [1] * len(fraud_scores)
                y_score = oos_scores + fraud_scores

                # AUCを計算
                auc = roc_auc_score(y_true, y_score)

                auc_results.append(
                    {
                        "period": period,
                        "method": method,
                        "auc": auc,
                        "n_oos": len(oos_scores),
                        "n_fraud": len(fraud_scores),
                    },
                )
            if len(oos_scores) > 0 and len(fraud_scores_amd) > 0:
                y_true = [0] * len(oos_scores) + [1] * len(fraud_scores_amd)
                y_score = oos_scores + fraud_scores_amd
                auc = roc_auc_score(y_true, y_score)
                auc_results.append(
                    {
                        "period": period,
                        "method": method,
                        "auc_amd": auc,
                        "n_oos": len(oos_scores),
                        "n_fraud_amd": len(fraud_scores_amd),
                    },
                )

    # 期間とメソッドごとのAUCをピボットテーブルで表示

    auc_results_df = pd.DataFrame(auc_results)
    auc_pivot = pd.pivot_table(
        auc_results_df,
        index="period",
        columns="method",
        values="auc",
    )
    return auc_pivot


def calc_auc_marco(rst_score):
    # OOSのスコアを取得（ラベル=0）
    oos_scores = (
        rst_score.query("period == @period and method == @method and task == 'oos'")
        .score_list.astype(float)
        .tolist()
    )

    # fraudのスコアを取得（ラベル=1）
    fraud_scores = (
        rst_score.query(
            "period == @period and method == @method and task == 'fraud'",
        )
        .score_list.astype(float)
        .tolist()
    )
    fraud_scores_amd = (
        rst_score.query(
            "period == @period and method == @method and task == 'fraud_amd'",
        )
        .score_list.astype(float)
        .tolist()
    )

    if len(oos_scores) > 0 and len(fraud_scores) > 0:
        # ラベルとスコアを結合
        y_true = [0] * len(oos_scores) + [1] * len(fraud_scores)
        y_score = oos_scores + fraud_scores

        # AUCを計算
        auc = roc_auc_score(y_true, y_score)


def calc_overall_auc(rst_score):
    # 年度ごとに同一年度内でROC曲線を計算し、ミクロ平均AUCとマクロ平均AUCを算出
    overall_auc_results = []
    mean_fpr = np.linspace(0, 1, 100)

    for method in rst_score["method"].unique():
        # --- fraud ---
        tprs_fraud = []
        aucs_fraud = []
        n_fraud_total = 0
        n_oos_total_fraud = 0

        for period in rst_score["period"].unique():
            oos_scores = (
                rst_score.query(
                    "method == @method and task == 'oos' and period == @period",
                )
                .score_list.astype(float)
                .tolist()
            )
            fraud_scores = (
                rst_score.query(
                    "method == @method and task == 'fraud' and period == @period",
                )
                .score_list.astype(float)
                .tolist()
            )
            if len(oos_scores) > 0 and len(fraud_scores) > 0:
                y_true = [0] * len(oos_scores) + [1] * len(fraud_scores)
                y_score = oos_scores + fraud_scores
                fpr, tpr, _ = roc_curve(y_true, y_score)
                tprs_fraud.append(np.interp(mean_fpr, fpr, tpr))
                aucs_fraud.append(roc_auc_score(y_true, y_score))
                n_fraud_total += len(fraud_scores)
                n_oos_total_fraud += len(oos_scores)

        if tprs_fraud:
            mean_tpr = np.mean(tprs_fraud, axis=0)
            mean_tpr[0] = 0.0
            mean_tpr[-1] = 1.0
            overall_auc = sklearn_auc(mean_fpr, mean_tpr)
            macro_auc = np.mean(aucs_fraud)
            overall_auc_results.append(
                {
                    "method": method,
                    "overall_auc": overall_auc,
                    "macro_auc": macro_auc,
                    "total_n_oos": n_oos_total_fraud,
                    "total_n_fraud": n_fraud_total,
                },
            )

        # --- fraud_amd ---
        tprs_amd = []
        aucs_amd = []
        n_amd_total = 0
        n_oos_total_amd = 0

        for period in rst_score["period"].unique():
            oos_scores = (
                rst_score.query(
                    "method == @method and task == 'oos' and period == @period",
                )
                .score_list.astype(float)
                .tolist()
            )
            fraud_scores_amd = (
                rst_score.query(
                    "method == @method and task == 'fraud_amd' and period == @period",
                )
                .score_list.astype(float)
                .tolist()
            )
            if len(oos_scores) > 0 and len(fraud_scores_amd) > 0:
                y_true = [0] * len(oos_scores) + [1] * len(fraud_scores_amd)
                y_score = oos_scores + fraud_scores_amd
                fpr, tpr, _ = roc_curve(y_true, y_score)
                tprs_amd.append(np.interp(mean_fpr, fpr, tpr))
                aucs_amd.append(roc_auc_score(y_true, y_score))
                n_amd_total += len(fraud_scores_amd)
                n_oos_total_amd += len(oos_scores)

        if tprs_amd:
            mean_tpr = np.mean(tprs_amd, axis=0)
            mean_tpr[0] = 0.0
            mean_tpr[-1] = 1.0
            overall_auc = sklearn_auc(mean_fpr, mean_tpr)
            macro_auc = np.mean(aucs_amd)
            overall_auc_results.append(
                {
                    "method": method,
                    "overall_auc_amd": overall_auc,
                    "macro_auc_amd": macro_auc,
                    "total_n_oos": n_oos_total_amd,
                    "total_n_fraud_amd": n_amd_total,
                },
            )

    overall_auc_df = pd.DataFrame(overall_auc_results)
    print("\n=== メソッドごとの全体AUC（全期間ミクロ平均・マクロ平均） ===")
    print(overall_auc_df)
    return overall_auc_df


def eval_model_restatement(
    n_topk: int = 5,
    N = None,  # noqa: N803
    random_state: int = 0,
) -> pd.DataFrame:
    results = []
    rng = np.random.default_rng(random_state)

    for period_end_dt_start, period_end_dt_end in [
        ("2020-04-01", "2021-03-31"),
        ("2021-04-01", "2022-03-31"),
        ("2022-04-01", "2023-03-31"),
        ("2023-04-01", "2024-03-31"),
        ("2024-04-01", "2025-03-31"),
    ]:
        amd_docid, amd_docid_error = get_task_div_pred_restatement(
            period_end_dt_start,
            period_end_dt_end,
        )
        print("amd_docid", len(amd_docid))
        print("amd_docid_error", len(amd_docid_error))
        calc_list = []
        common_fraud_set = set(amd_docid_error)
        common_amd_set = set(amd_docid)
        common_oos_set = None
        for method_name in cfg["method_name_list"]:
            print(f"===== method_name: {method_name} =====")
            distance_oos = load_distance_oos(
                period_end_dt_start,
                period_end_dt_end,
                method=method_name,
            )
            distance_oos["dist"] = 1 - distance_oos["dist"]

            distance_fraud = load_distance_restatement(
                period_end_dt_start,
                period_end_dt_end,
                method=method_name,
            )
            print(
                "error not calculated:",
                set(amd_docid_error) - set(distance_fraud.source_docid),
            )

            print("distance_error", len(distance_fraud))
            distance_fraud["dist"] = 1 - distance_fraud["dist"]
            distance_fraud_amd = load_distance_restatement_amd(
                period_end_dt_start,
                period_end_dt_end,
                method=method_name,
            )
            print(
                "amd not calculated:",
                set(amd_docid) - set(distance_fraud_amd.source_docid),
            )
            distance_fraud_amd["dist"] = 1 - distance_fraud_amd["dist"]
            calc_list.append(
                {
                    "method_name": method_name,
                    "distance_df_oos": distance_oos,
                    "distance_df_fraud": distance_fraud,
                    "distance_df_fraud_amd": distance_fraud_amd,
                },
            )
            common_fraud_set = set(distance_fraud.source_docid) & common_fraud_set
            common_amd_set = set(distance_fraud_amd.source_docid) & common_amd_set
            distance_oos_set = set(distance_oos.source_docid)
            common_oos_set = (
                distance_oos_set
                if common_oos_set is None
                else distance_oos_set & common_oos_set
            )

        # correlation (already converted to distance)
        print("===== correlation =====")
        cor_distance_fraud = load_distance_restatement2(
            period_end_dt_start,
            period_end_dt_end,
            method="correlation",
        )
        print(
            "fraud not calculated:",
            set(amd_docid_error) - set(cor_distance_fraud.source_docid),
        )
        cor_distance_fraud_amd = load_distance_restatement_amd2(
            period_end_dt_start,
            period_end_dt_end,
            method="correlation",
        )
        print(
            "amd not calculated:",
            set(amd_docid) - set(cor_distance_fraud_amd.source_docid),
        )
        cor_distance_oos = load_distance_oos(
            period_end_dt_start,
            period_end_dt_end,
            method="correlation",
        )
        calc_list.append(
            {
                "method_name": "correlation",
                "distance_df_oos": cor_distance_oos,
                "distance_df_fraud": cor_distance_fraud,
                "distance_df_fraud_amd": cor_distance_fraud_amd,
            },
        )
        common_fraud_set = set(cor_distance_fraud.source_docid) & common_fraud_set
        common_amd_set = set(cor_distance_fraud_amd.source_docid) & common_amd_set
        distance_oos_set = set(cor_distance_oos.source_docid)
        common_oos_set = (
            distance_oos_set
            if common_oos_set is None
            else distance_oos_set & common_oos_set
        )
        # mahalanobis
        print("===== mahalanobis =====")
        mahalanobis_distance_fraud = load_distance_restatement2(
            period_end_dt_start,
            period_end_dt_end,
            method="mahalanobis",
        )
        print(
            "fraud not calculated:",
            set(amd_docid_error) - set(mahalanobis_distance_fraud.source_docid),
        )
        # mahalanobis_distance_fraud["dist"] = 1 - mahalanobis_distance_fraud["dist"]
        mahalanobis_distance_fraud_amd = load_distance_restatement_amd2(
            period_end_dt_start,
            period_end_dt_end,
            method="mahalanobis",
        )
        print(
            "amd not calculated:",
            set(amd_docid) - set(mahalanobis_distance_fraud_amd.source_docid),
        )
        # mahalanobis_distance_fraud_amd["dist"] = (
        #    1 - mahalanobis_distance_fraud_amd["dist"]
        # )
        mahalanobis_distance_oos = load_distance_oos2(
            period_end_dt_start,
            period_end_dt_end,
            method="mahalanobis",
        )
        # mahalanobis_distance_oos["dist"] = 1 - mahalanobis_distance_oos["dist"]

        calc_list.append(
            {
                "method_name": "mahalanobis",
                "distance_df_fraud": mahalanobis_distance_fraud,
                "distance_df_fraud_amd": mahalanobis_distance_fraud_amd,
                "distance_df_oos": mahalanobis_distance_oos,
            },
        )
        common_fraud_set = (
            set(mahalanobis_distance_fraud.source_docid) & common_fraud_set
        )
        common_amd_set = (
            set(mahalanobis_distance_fraud_amd.source_docid) & common_amd_set
        )
        distance_oos_set = set(mahalanobis_distance_oos.source_docid)
        common_oos_set = (
            distance_oos_set
            if common_oos_set is None
            else distance_oos_set & common_oos_set
        )
        # BM25
        print("=== BM25 === ")
        bm25_distance_fraud = load_distance_restatement2(
            period_end_dt_start,
            period_end_dt_end,
            method="bm25",
        )
        print(
            "fraud not calculated:",
            set(amd_docid_error) - set(bm25_distance_fraud.source_docid),
        )
        # bm25_distance_fraud["dist"] = 1 - bm25_distance_fraud["dist"]
        bm25_distance_fraud_amd = load_distance_restatement_amd2(
            period_end_dt_start,
            period_end_dt_end,
            method="bm25",
        )
        print(
            "amd not calculated:",
            set(amd_docid) - set(bm25_distance_fraud_amd.source_docid),
        )
        # bm25_distance_fraud_amd["dist"] = 1 - bm25_distance_fraud_amd["dist"]
        bm25_distance_oos = load_distance_oos2(
            period_end_dt_start,
            period_end_dt_end,
            method="bm25",
        )
        # bm25_distance_oos["dist"] = 1 - bm25_distance_oos["dist"]
        calc_list.append(
            {
                "method_name": "bm25",
                "distance_df_fraud": bm25_distance_fraud,
                "distance_df_fraud_amd": bm25_distance_fraud_amd,
                "distance_df_oos": bm25_distance_oos,
            },
        )
        common_fraud_set = set(bm25_distance_fraud.source_docid) & common_fraud_set
        common_amd_set = set(bm25_distance_fraud_amd.source_docid) & common_amd_set
        distance_oos_set = set(bm25_distance_oos.source_docid)
        common_oos_set = (
            distance_oos_set
            if common_oos_set is None
            else distance_oos_set & common_oos_set
        )
        if N is None:
            eval_oos_set = None
        else:
            if N <= 0:
                msg = "N must be positive or None"
                raise ValueError(msg)
            n_oos_sample = int(np.ceil(len(common_fraud_set) * N))
            n_oos_sample = min(n_oos_sample, len(common_oos_set))
            eval_oos_set = set(
                rng.choice(
                    sorted(common_oos_set),
                    size=n_oos_sample,
                    replace=False,
                ),
            )
            print("common_oos_set", len(common_oos_set))
            print("eval_oos_set", len(eval_oos_set), f"(N={N})")
        for calc in calc_list:
            results.append(
                {
                    "period": f"{period_end_dt_start}_{period_end_dt_end}",
                    "method": calc["method_name"],
                    "task": "fraud",
                    "score_list": calc["distance_df_fraud"]
                    .query("source_docid in @common_fraud_set")
                    .groupby("source_docid")
                    .apply(agg_func_kNN_distance, n_topk=n_topk, col_name="dist")
                    .to_list(),
                },
            )
            results.append(
                {
                    "period": f"{period_end_dt_start}_{period_end_dt_end}",
                    "method": calc["method_name"],
                    "task": "fraud_amd",
                    "score_list": calc["distance_df_fraud_amd"]
                    .query("source_docid in @common_amd_set")
                    .groupby("source_docid")
                    .apply(agg_func_kNN_distance, n_topk=n_topk, col_name="dist")
                    .to_list(),
                },
            )
            results.append(
                {
                    "period": f"{period_end_dt_start}_{period_end_dt_end}",
                    "method": calc["method_name"],
                    "task": "oos",
                    "score_list": (
                        calc["distance_df_oos"]
                        if eval_oos_set is None
                        else calc["distance_df_oos"].query(
                            "source_docid in @eval_oos_set",
                        )
                    )
                    .groupby("source_docid")
                    .apply(agg_func_kNN_distance, n_topk=n_topk, col_name="dist")
                    .to_list(),
                },
            )

    rst_score = pd.DataFrame(results).explode("score_list")
    return rst_score


# %%

# %%

RESULTSDIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/results")


def main() -> None:
    for n_topk in [5, 20]:
        rst_score_fraud = eval_model_fraud(n_topk=n_topk, N=10)
        rst_score_fraud.query("method == 'clip_base' and task == 'oos'").score_list.hist(
            bins=20,
        )
        rst_score_fraud.query(
            "method == 'clip_base' and task == 'fraud'",
        ).score_list.hist(
            bins=20,
        )

        rst_score_fraud.to_pickle(
            RESULTSDIR / "results_tbl" / f"results_fraud_{cfg['experiment_name']}_{n_topk}.pkl",
        )

        out_fn = RESULTSDIR / "results_tbl" / f"overall_auc_fraud_{cfg['experiment_name']}_{n_topk}.pkl"
        overall_auc_df = calc_overall_auc(rst_score_fraud)
        overall_auc_df.to_pickle(out_fn)

    for n_topk in [5, 20]:
        rst_score_restatement = eval_model_restatement(n_topk=n_topk, N=10)
        rst_score_restatement.to_pickle(
            RESULTSDIR / "results_tbl" / f"results_restatement_{cfg['experiment_name']}_{n_topk}.pkl",
        )

        overall_auc_df = calc_overall_auc(rst_score_restatement)
        overall_auc_df.to_pickle(
            RESULTSDIR / "results_tbl" / f"overall_auc_restatement_{cfg['experiment_name']}_{n_topk}.pkl",
        )


if __name__ == "__main__":
    main()