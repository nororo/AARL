# %%
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

XBRL_PROJPATH = r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
XBRL_PROJDIR = Path(XBRL_PROJPATH)
PROCDIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/data/3_processed")
DATADIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/data/1_raw")
INTERMEDIATEDIR = Path(
    "/Users/noro/Documents/Projects/t_interpretable_fs/data/2_intermediate",
)
RESULTSDIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/results")
# %% Business Sector #########################################################
# Get business sector from EdinetcodeDlInfo
#########################################################
import sys

sys.path.append(r"/Users/noro/Documents/Projects/t_interpretable_fs")
import yaml

CFGDIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/src")
cfg = yaml.load(open(CFGDIR / "cfg_exp_add.yaml"), Loader=yaml.FullLoader)
from src.libs.downstream_task import (
    get_eval_docid_spec_list,
    get_sector,
    get_task_div_pred_sector,
)


# %% pred model
def load_ot_distance_list_sector() -> pd.DataFrame:
    filename = PROCDIR / "distance" / "ot_distance_list_sector.pkl"
    ot_distance_list_sector = joblib.load(filename)
    ot_distance_list_df_sector = pd.DataFrame(ot_distance_list_sector)
    ot_distance_list_df_sector = ot_distance_list_df_sector.query(
        "bs_ot_distance.notna()",
    )
    ot_distance_list_df_sector = ot_distance_list_df_sector.assign(
        ot_distance=ot_distance_list_df_sector.bs_ot_distance
        + ot_distance_list_df_sector.pl_ot_distance,
    )
    return ot_distance_list_df_sector


def load_mahalanobis_distance_sector() -> pd.DataFrame:
    filename = INTERMEDIATEDIR / "distance" / "mahalanobis" / "distance_sector.pkl"
    mahalanobis_distance_sector = pd.read_pickle(filename)
    return mahalanobis_distance_sector


def load_cor_distance_sector() -> pd.DataFrame:
    filename = INTERMEDIATEDIR / "distance" / "correlation" / "distance_sector.pkl"
    cor_distance_sector = pd.read_pickle(filename)
    return cor_distance_sector


def load_distance_sector(method: str = "clip") -> pd.DataFrame:
    filename = INTERMEDIATEDIR / "distance" / method / "distance_sector.pkl"
    raw = pd.read_pickle(filename)
    print(f"  [load_distance_sector] method={method}")
    print(f"  [load_distance_sector] file: {filename}")
    print(f"  [load_distance_sector] raw shape: {raw.shape}")
    print(f"  [load_distance_sector] raw columns: {raw.columns.tolist()}")
    if len(raw) > 0:
        print(f"  [load_distance_sector] raw source sample: {raw.iloc[:3, 0].tolist()}")
        print(f"  [load_distance_sector] raw target sample: {raw.iloc[:3, 1].tolist()}")
    distance_sector = raw.rename(
        columns={"docid_source": "source_docid", "docid_target": "target_docid"},
    )
    print(
        f"  [load_distance_sector] after rename columns: {distance_sector.columns.tolist()}",
    )
    before_filter = len(distance_sector)
    distance_sector = distance_sector.query("source_docid != target_docid")
    print(
        f"  [load_distance_sector] after self-pair filter: {before_filter} -> {len(distance_sector)}",
    )
    return distance_sector


class EvalDistance:
    def __init__(self, exclude_docid_spec: bool = False):
        # sector
        # self.distance_df = distance_df
        (
            eval_sector_test,
            retrieval_pool_sector,
        ) = get_task_div_pred_sector(
            exclude_docid_spec_list=exclude_docid_spec,
        )  # sector pred
        self.eval_sector_test = eval_sector_test
        self.retrieval_pool_sector = retrieval_pool_sector
        # kpi

    def eval_distance_sector(self, distance_df, col_name="ot_distance", n_topk=20):
        """distance_df: pd.DataFrame
        distance_df.columns: ["source_docid", "target_docid", "ot_distance"]
        """
        # デバッグ: フィルタリング前の情報
        print(f"  [before filter] distance_df.shape: {distance_df.shape}")
        print(f"  [before filter] distance_df.columns: {distance_df.columns.tolist()}")
        print(
            f"  [before filter] source_docid sample: {distance_df['source_docid'].unique()[:3]}, type: {type(distance_df['source_docid'].iloc[0]) if len(distance_df) > 0 else 'N/A'}",
        )
        print(
            f"  [before filter] target_docid sample: {distance_df['target_docid'].unique()[:3]}, type: {type(distance_df['target_docid'].iloc[0]) if len(distance_df) > 0 else 'N/A'}",
        )
        print(
            f"  [eval] eval_sector_test.index sample: {list(self.eval_sector_test.index[:3])}, type: {type(self.eval_sector_test.index[0])}",
        )
        print(
            f"  [pool] retrieval_pool.index sample: {list(self.retrieval_pool_sector.index[:3])}, type: {type(self.retrieval_pool_sector.index[0])}",
        )
        # source_docid と eval_sector_test.index の重なりを確認
        src_overlap = set(distance_df["source_docid"].unique()) & set(
            self.eval_sector_test.index,
        )
        tgt_overlap = set(distance_df["target_docid"].unique()) & set(
            self.retrieval_pool_sector.index,
        )
        print(
            f"  [overlap] source ∩ eval: {len(src_overlap)}, target ∩ pool: {len(tgt_overlap)}",
        )

        distance_df = distance_df.query(
            "source_docid in @self.eval_sector_test.index and target_docid in @self.retrieval_pool_sector.index",
        )
        print(f"  [after filter] distance_df.shape: {distance_df.shape}")
        pred_sector = distance_df.groupby("source_docid").apply(
            self.agg_func_kNN,
            n_topk=n_topk,
            col_name=col_name,
            task="sector",
        )
        # display(pred_sector)

        # groupby().apply() は空のDataFrameに対してはDataFrameを返すため、
        # SeriesかDataFrameかに応じて処理を分岐する
        if isinstance(pred_sector, pd.Series):
            pred_sector_df = pred_sector.to_frame(name="pred_sector")
        elif isinstance(pred_sector, pd.DataFrame):
            if pred_sector.empty:
                pred_sector_df = pd.DataFrame(columns=["pred_sector"])
            else:
                pred_sector_df = pred_sector.rename(
                    columns={pred_sector.columns[0]: "pred_sector"},
                )
        else:
            pred_sector_df = pd.DataFrame(columns=["pred_sector"])

        eval_sector_test_pred = pd.merge(
            self.eval_sector_test,
            pred_sector_df,
            left_index=True,
            right_index=True,
            how="left",
        )

        score = eval_sector_test_pred.query("pred_sector == business_class_tse").shape[
            0
        ] / len(
            self.eval_sector_test,
        )
        return score

    def agg_func_kNN(self, sr, n_topk=5, col_name="ot_distance", task="sector") -> str:
        sr = sr.sort_values(by=col_name)
        topk_list = sr.iloc[:n_topk].target_docid.to_list()
        if task == "sector":
            topk_list = [
                self.retrieval_pool_sector.loc[docid, "business_class_tse"]
                for docid in topk_list
            ]
        elif task == "kpi":
            topk_list = [
                self.retrieval_val_kpi.loc[docid, "kpi_diff_rate_bin"]
                for docid in topk_list
            ]
        else:
            raise ValueError(f"Invalid task: {task}")
        return pd.Series(topk_list).value_counts().index[0]

    def agg_func_kNN_test(self, sr, n_topk=5, col_name="ot_distance"):
        sr = sr.sort_values(by=col_name)
        topk_list = sr.iloc[:n_topk].target_edinetcode.to_list()

        return topk_list


# %%
def get_results_sector():
    response_tbl_with_sector_test, response_tbl_with_sector_train = get_sector()
    response_tbl_with_sector_test.year.value_counts()
    response_tbl_with_sector_test.query(
        "period_end_dt >= '2024-04-01' and period_end_dt <= '2025-03-31'",
    )  # .response_edinetCode.nunique()
    response_tbl_with_sector_train.year.value_counts()
    response_tbl_with_sector_train.query(
        "period_end_dt >= '2024-04-01' and period_end_dt <= '2025-03-31'",
    )  # .response_edinetCode.nunique()

    (
        eval_sector_test,
        retrieval_pool_sector,
    ) = get_task_div_pred_sector()  # sector pred

    print("load_clip_distance_sector")

    calc_list = []
    for method_name in cfg["method_name_list"]:
        distance_sector = load_distance_sector(method=method_name)
        distance_sector["dist"] = 1 - distance_sector["dist"]
        calc_list.append(
            {
                "method_name": method_name,
                "distance_df_sector": distance_sector,
            },
        )
    print("load_cor_distance_sector")
    # corelation
    cor_distance_sector = load_cor_distance_sector()
    calc_list.append(
        {
            "method_name": "correlation",
            "distance_df_sector": cor_distance_sector,
        },
    )
    # mahalanobis
    mahalanobis_distance_sector = load_mahalanobis_distance_sector()
    calc_list.append(
        {
            "method_name": "mahalanobis",
            "distance_df_sector": mahalanobis_distance_sector,
        },
    )
    # bm25
    distance_sector = load_distance_sector(
        method="bm25",
    )  # already transformed to distance
    calc_list.append(
        {
            "method_name": "bm25",
            "distance_df_sector": distance_sector,
        },
    )

    results = []
    n_topk_sector = cfg["n_topk_sector"]
    for calc in calc_list:
        eval_docid_spec_list = get_eval_docid_spec_list(
            test_docid=eval_sector_test.index.tolist(),
        )

        results.append(
            {
                "method": calc["method_name"],
                "task": "pred_sector",
                "score": EvalDistance(exclude_docid_spec=True).eval_distance_sector(
                    calc["distance_df_sector"],
                    col_name="dist",
                    n_topk=n_topk_sector,
                ),
            },
        )
        results.append(
            {
                "method": calc["method_name"],
                "task": "pred_sector_all",
                "score": EvalDistance(exclude_docid_spec=False).eval_distance_sector(
                    calc["distance_df_sector"],
                    col_name="dist",
                    n_topk=n_topk_sector,
                ),
            },
        )

    results_df = pd.pivot_table(
        pd.DataFrame(results),
        index="method",
        columns="task",
        values="score",
        aggfunc=np.mean,
    )
    return results_df


def main() -> pd.DataFrame:
    results_df = get_results_sector()
    out_fn = RESULTSDIR / "results_tbl" / f"results_sector_{cfg['experiment_name']}_{cfg['n_topk_sector']}.pkl"
    results_df.to_pickle(out_fn)
    return results_df

# %%
if __name__ == "__main__":
    main()
# %%
results_df = get_results_sector()
# (
#    eval_sector_test,
#    retrieval_pool_sector,
# ) = get_task_div_pred_sector()  # sector pred
# %%
results_df

# %%
#filename_business_class = "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/0_metadata/dataset_2507/tiba/tse_sector_2025-03-31.csv"
#business_class = pd.read_csv(
#    filename_business_class,
#    header=0,
#    index_col=0,
#    dtype={"sec_code": str},
#)
#business_class["date_ts"] = pd.to_datetime(business_class["date"])
#business_class
#
#
## %%
#business_class_g = business_class.groupby(by="sec_code")
#business_class_latest = business_class.loc[
#    business_class_g["date_ts"].idxmax(),
#    :,
#]