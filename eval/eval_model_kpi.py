# %%
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.append(r"./Projects/t_interpretable_fs")
import yaml

CFGDIR = Path("./Projects/t_interpretable_fs/src")

XBRL_PROJPATH = r"./Projects/XBRL_common_space_projection/"
XBRL_PROJDIR = Path(XBRL_PROJPATH)
PROCDIR = Path("./Projects/t_interpretable_fs/data/3_processed")
INTERMEDIATEDIR = Path(
    "./Projects/t_interpretable_fs/data/2_intermediate",
)
RESULTSDIR = Path("./Projects/t_interpretable_fs/results")
from libs.downstream_task import get_task_div_pred_kpi

DATADIR = Path("./Projects/t_interpretable_fs/data/1_raw")

cfg = yaml.load(open(CFGDIR / "cfg_exp_add.yaml"), Loader=yaml.FullLoader)


def get_fiscal_quarter(period_end_dt):
    """決算日から決算月のクォーターを取得

    Parameters
    ----------
    period_end_dt : datetime
        決算日

    Returns
    -------
    str
        クォーター ('Q1': 1-3月, 'Q2': 4-6月, 'Q3': 7-9月, 'Q4': 10-12月)

    """
    month = period_end_dt.month
    if 1 <= month <= 3:
        return "Q1"
    if 4 <= month <= 6:
        return "Q2"
    if 7 <= month <= 9:
        return "Q3"
    # 10-12月
    return "Q4"


# train 2020/12/31
# eval 2021
# %%
train_periodend_startdate = cfg["train_periodend_startdate"]
train_periodend_enddate = cfg["train_periodend_enddate"]
eval_periodend_startdate = cfg["eval_periodend_startdate"]
eval_periodend_enddate = cfg["eval_periodend_enddate"]
train_periodend_startdate

# %%
"""
訓練評価データのXとYは異なるterm（eval term）なので、docIDリストはそれぞれ2つ必要
"""


# %% pred model


# %%
def load_ot_distance_list_kpi() -> pd.DataFrame:
    filename = PROCDIR / "distance" / "ot_distance_list_kpi.pkl"
    ot_distance_list_kpi = joblib.load(filename)
    ot_distance_list_df_kpi = pd.DataFrame(ot_distance_list_kpi)
    ot_distance_list_df_kpi = ot_distance_list_df_kpi.query(
        "bs_ot_distance.notna()",
    )
    ot_distance_list_df_kpi = ot_distance_list_df_kpi.assign(
        ot_distance=ot_distance_list_df_kpi.bs_ot_distance
        + ot_distance_list_df_kpi.pl_ot_distance,
    )
    return ot_distance_list_df_kpi


def load_ot_distance_list_kpi_2020() -> pd.DataFrame:
    filename = PROCDIR / "distance" / "ot_distance_list_kpi_2020.pkl"
    ot_distance_list = joblib.load(filename)
    ot_distance_list_df = pd.DataFrame(ot_distance_list)
    ot_distance_list_df = ot_distance_list_df.query(
        "bs_ot_distance.notna()",
    )
    ot_distance_list_df = ot_distance_list_df.assign(
        ot_distance=ot_distance_list_df.bs_ot_distance
        + ot_distance_list_df.pl_ot_distance,
    )
    return ot_distance_list_df


def load_cor_distance_kpi() -> pd.DataFrame:
    filename = INTERMEDIATEDIR / "distance" / "correlation" / "distance_kpi.pkl"
    cor_distance_kpi = pd.read_pickle(filename).query("source_docid != target_docid")
    return cor_distance_kpi


def load_mahalanobis_distance_kpi() -> pd.DataFrame:
    filename = INTERMEDIATEDIR / "distance" / "mahalanobis" / "distance_kpi.pkl"
    mahalanobis_distance_kpi = pd.read_pickle(filename).query(
        "source_docid != target_docid",
    )
    return mahalanobis_distance_kpi


def load_distance_kpi(method: str = "clip") -> pd.DataFrame:
    filename = INTERMEDIATEDIR / "distance" / method / "distance_kpi.pkl"
    distance_kpi = (
        pd.read_pickle(filename)
        .rename(
            columns={"docid_source": "source_docid", "docid_target": "target_docid"},
        )
        .query("source_docid != target_docid")
    )
    return distance_kpi


def load_clip_distance_kpi_1104() -> pd.DataFrame:
    filename = PROCDIR / "distance_clip" / "distance_kpi_1104.pkl"
    clip_distance_kpi = (
        pd.read_pickle(filename)
        .rename(
            columns={"docid_source": "source_docid", "docid_target": "target_docid"},
        )
        .query("source_docid != target_docid")
    )
    return clip_distance_kpi


def get_similar_companies_by_kpi_trend(
    target_edinet_code,
    kpi_val_test_pivot,
    top_k=5,
    method="correlation",
):
    """指定された企業のKPI年次推移パターンに類似した企業トップKを取得する

    Parameters
    ----------
    target_edinet_code : str
        対象企業のEDINETコード
    top_k : int
        取得する類似企業数 (default: 5)
    method : str
        類似度計算方法 ("correlation", "cosine", "euclidean")

    Returns
    -------
    pd.Series
        類似度が高い順の企業リスト（類似度スコア付き）

    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

    if target_edinet_code not in kpi_val_test_pivot.index:
        print(f"企業コード {target_edinet_code} がデータに存在しません")
        return None

    target_trend = kpi_val_test_pivot.loc[target_edinet_code].values.reshape(1, -1)

    if method == "correlation":
        # 相関係数による類似度
        correlations = []
        for edinet_code in kpi_val_test_pivot.index:
            if edinet_code != target_edinet_code:
                other_trend = kpi_val_test_pivot.loc[edinet_code].values.reshape(1, -1)
                # NaNがある場合の処理
                if not (np.isnan(target_trend).any() or np.isnan(other_trend).any()):
                    corr = np.corrcoef(target_trend.flatten(), other_trend.flatten())[
                        0,
                        1,
                    ]
                    correlations.append((edinet_code, corr))

        # 相関係数の高い順にソート
        correlations.sort(key=lambda x: x[1], reverse=True)
        result = pd.Series(
            [x[1] for x in correlations[:top_k]],
            index=[x[0] for x in correlations[:top_k]],
        )

    elif method == "cosine":
        # コサイン類似度
        all_trends = kpi_val_test_pivot.values
        similarities = cosine_similarity(target_trend, all_trends)[0]

        # 自分自身を除外
        mask = kpi_val_test_pivot.index != target_edinet_code
        filtered_similarities = similarities[mask]
        filtered_indices = kpi_val_test_pivot.index[mask]

        # 類似度の高い順にソート
        top_indices = np.argsort(filtered_similarities)[::-1][:top_k]
        result = pd.Series(
            filtered_similarities[top_indices],
            index=filtered_indices[top_indices],
        )

    elif method == "euclidean":
        # ユークリッド距離（小さいほど類似）
        all_trends = kpi_val_test_pivot.values
        distances = euclidean_distances(target_trend, all_trends)[0]

        # 自分自身を除外
        mask = kpi_val_test_pivot.index != target_edinet_code
        filtered_distances = distances[mask]
        filtered_indices = kpi_val_test_pivot.index[mask]

        # 距離の小さい順にソート
        top_indices = np.argsort(filtered_distances)[:top_k]
        result = pd.Series(
            filtered_distances[top_indices],
            index=filtered_indices[top_indices],
        )

    return result


def load_kpi_val() -> pd.DataFrame:
    filename = DATADIR / "preproc_log.pkl"
    preproc_log = pd.read_pickle(filename)
    preproc_log

    filename = DATADIR / "kpi_val.pkl"
    kpi_val = pd.read_pickle(filename)
    kpi_val.index = preproc_log.docid
    kpi_val = kpi_val[["cy_pl_val", "py_pl_val", "cy_kpi_val", "py_kpi_val"]]
    kpi_val = kpi_val.assign(
        revinue_diff=kpi_val.cy_pl_val - kpi_val.py_pl_val,
        revinue_diff_rate=(kpi_val.cy_pl_val - kpi_val.py_pl_val)
        / kpi_val.py_pl_val.abs(),
        kpi_diff=kpi_val.cy_kpi_val - kpi_val.py_kpi_val,
        kpi_diff_rate=(kpi_val.cy_kpi_val - kpi_val.py_kpi_val)
        / kpi_val.py_kpi_val.abs(),
    )
    kpi_val["kpi_diff_rate_bin"] = pd.cut(
        kpi_val.kpi_diff_rate.clip(-0.5, 0.5),
        bins=[-0.6, -0.1, 0, 0.1, 0.6],
    )
    kpi_val["revinue_diff_rate_bin"] = pd.cut(
        kpi_val.revinue_diff_rate.clip(-0.5, 0.5),
        bins=[-0.6, -0.1, 0, 0.1, 0.6],
    )
    return kpi_val


class EvalDistance:
    def __init__(self, pred_target_name: str = "kpi_diff_rate"):
        # sector
        # self.distance_df = distance_df

        # kpi
        (
            eval_kpi_test_df,
            retrieval_pool_kpi_df,
            eval_kpi_test_5,
            retrieval_pool_kpi_5,
        ) = get_task_div_pred_kpi(
            eval_periodend_startdate="2020-04-01",
            eval_periodend_enddate="2025-03-31",
            train_periodend_startdate="2014-03-31",
            train_periodend_enddate="2020-03-31",
        )  # kpi pred
        self.eval_kpi_test = eval_kpi_test_df
        self.retrieval_pool_kpi = retrieval_pool_kpi_df
        self.eval_kpi_test_5 = eval_kpi_test_5
        self.retrieval_pool_kpi_5 = retrieval_pool_kpi_5
        self.pred_target_name = pred_target_name

    def eval_distance_kpi(
        self,
        distance_df,
        col_name="ot_distance",
        target_year="2022",
        n_topk=20,
        pred_kpi_flg: bool = True,
    ):
        """distance_df: pd.DataFrame
        distance_df.columns: ["source_docid", "target_docid", "ot_distance"]
        """
        distance_df = distance_df.query(
            "source_docid in @self.eval_kpi_test.index and target_docid in @self.retrieval_pool_kpi.index",
        )
        print(distance_df.shape)
        kpi_val = load_kpi_val()
        kpi_val_test = kpi_val.query("index in @self.eval_kpi_test.index")
        kpi_val_test["response_edinetCode"] = self.eval_kpi_test["response_edinetCode"]
        kpi_val_test["period_end_dt"] = self.eval_kpi_test["period_end_dt"]

        # 会計年度を計算（3/31ベース: 2020/4-2021/3 -> '2021'）
        # period_end_dt（決算日）が2021/3/31なら'2021'年度、2021/4/1以降なら'2022'年度
        kpi_val_test["fiscal_year"] = kpi_val_test["period_end_dt"].apply(
            lambda x: str(x.year + 1) if x.month >= 4 else str(x.year),
        )

        kpi_val_retrieval = kpi_val.query("index in @self.retrieval_pool_kpi_5.index")
        self.retrieval_val_kpi = kpi_val_retrieval

        pred_kpi = distance_df.groupby("source_docid").apply(
            self.agg_func_kNN,
            n_topk=n_topk,
            col_name=col_name,
            task="kpi",
        )
        eval_kpi_test_pred = pd.merge(
            kpi_val_test,
            pred_kpi.to_frame(name="pred_kpi"),
            left_index=True,
            right_index=True,
            how="left",
        )
        # evaluation split
        kpi_val_test_5 = kpi_val.query("index in @self.eval_kpi_test_5.index")
        kpi_val_test_5["response_edinetCode"] = self.eval_kpi_test_5[
            "response_edinetCode"
        ]
        kpi_val_test_5["period_end_dt"] = self.eval_kpi_test_5["period_end_dt"]

        # 会計年度を計算（3/31ベース: 2020/4-2021/3 -> '2021'）
        # period_end_dt（決算日）が2021/3/31なら'2021'年度、2021/4/1以降なら'2022'年度
        kpi_val_test_5["fiscal_year"] = kpi_val_test_5["period_end_dt"].apply(
            lambda x: str(x.year + 1) if x.month >= 4 else str(x.year),
        )
        kpi_val_test_5_next = kpi_val_test_5.query("fiscal_year == @target_year")

        eval_kpi_test_pred = eval_kpi_test_pred.merge(
            kpi_val_test_5_next[["response_edinetCode", "kpi_diff_rate_bin"]].rename(
                columns={
                    "response_edinetCode": "target_edinetcode",
                    "kpi_diff_rate_bin": "target_kpi_diff_rate_bin",
                    "revinue_diff_rate_bin": "target_revinue_diff_rate_bin",
                },
            ),
            left_on="response_edinetCode",
            right_on="target_edinetcode",
            how="left",
        )

        if pred_kpi_flg:
            score = eval_kpi_test_pred.query(
                "pred_kpi == target_kpi_diff_rate_bin",
            ).shape[0] / len(self.eval_kpi_test)
        else:
            score = eval_kpi_test_pred.query(
                "pred_kpi == target_revinue_diff_rate_bin",
            ).shape[0] / len(self.eval_kpi_test)
        return score

    def eval_distance_kpi_trend(
        self,
        distance_df,
        col_name="ot_distance",
        ans_top_k=5,
        pred_top_k=20,
    ):
        """distance_df: pd.DataFrame
        distance_df.columns: ["source_edinetcode", "target_edinetcode", "ot_distance"]
        """
        kpi_val = load_kpi_val()
        kpi_val_test = kpi_val.query("index in @self.eval_kpi_test_5.index")
        kpi_val_test["response_edinetCode"] = self.eval_kpi_test_5[
            "response_edinetCode"
        ]
        kpi_val_test["period_end_dt"] = self.eval_kpi_test_5["period_end_dt"]

        # 会計年度を計算（3/31ベース: 2020/4-2021/3 -> '2021'）
        # period_end_dt（決算日）が2021/3/31なら'2021'年度、2021/4/1以降なら'2022'年度
        kpi_val_test["fiscal_year"] = kpi_val_test["period_end_dt"].apply(
            lambda x: str(x.year + 1) if x.month >= 4 else str(x.year),
        )
        # self.kpi_val_test = kpi_val_test
        print(len(kpi_val_test))

        kpi_val_test["pred_target_clip"] = kpi_val_test[self.pred_target_name].clip(
            -0.5,
            0.5,
        )
        kpi_val_test_pivot = pd.pivot_table(
            kpi_val_test.query("fiscal_year in ['2021','2022','2023','2024','2025']"),
            index="response_edinetCode",
            columns="fiscal_year",
            values="pred_target_clip",
            aggfunc=np.mean,
        ).fillna(0)

        ans_list = []
        ans_dict_list = []
        for edinet_code in kpi_val_test_pivot.index:
            similar_companies_euclidean = get_similar_companies_by_kpi_trend(
                edinet_code,
                kpi_val_test_pivot,
                top_k=ans_top_k,
                method="euclidean",
            )
            ans_list.append(
                {
                    "edinet_code": edinet_code,
                    "similar_companies": list(similar_companies_euclidean.index),
                },
            )
            ans_dict_list.append(
                {
                    "edinet_code": edinet_code,
                    "similar_companies": similar_companies_euclidean.to_dict(),
                },
            )

        pred_kpi = distance_df.groupby("source_edinetcode").apply(
            self.agg_func_kNN_test,
            n_topk=pred_top_k,
            col_name=col_name,
        )

        eval_kpi_test_pred = pd.merge(
            pd.DataFrame(ans_list).set_index("edinet_code"),
            pred_kpi.to_frame(name="pred_kpi"),
            left_index=True,
            right_index=True,
            how="inner",
        )
        score = eval_kpi_test_pred.apply(
            lambda x: len(set(x.pred_kpi) & set(x.similar_companies)) / len(x.pred_kpi),
            axis=1,
        ).sum() / len(eval_kpi_test_pred)
        return score

    def agg_func_kNN(self, sr, n_topk=5, col_name="ot_distance", task="sector"):
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

    def eval_distance_kpi_trend_map(
        self,
        distance_df,
        col_name="ot_distance",
        ans_top_k=5,
        filter_by_quarter=False,
    ):
        """正解数kを与えた時のMean Average Precision (MAP@k)を計算する

        Parameters
        ----------
        distance_df : pd.DataFrame
            distance_df.columns: ["source_edinetcode", "target_edinetcode", col_name]
        col_name : str
            距離のカラム名
        ans_top_k : int
            正解数k (KPI年次推移でユークリッド距離が近いトップk社を正解とする)
        filter_by_quarter : bool, default False
            Trueの場合、決算期（クォーター）が同じ企業同士でのみ検索を行う

        Returns
        -------
        float
            MAP@k score

        """
        kpi_val = load_kpi_val()

        # test企業のKPI推移
        kpi_val_test = kpi_val.query("index in @self.eval_kpi_test_5.index")
        kpi_val_test["response_edinetCode"] = self.eval_kpi_test_5[
            "response_edinetCode"
        ]
        kpi_val_test["period_end_dt"] = self.eval_kpi_test_5["period_end_dt"]

        # 会計年度を計算（3/31ベース: 2020/4-2021/3 -> '2021'）
        # period_end_dt（決算日）が2021/3/31なら'2021'年度、2021/4/1以降なら'2022'年度
        kpi_val_test["fiscal_year"] = kpi_val_test["period_end_dt"].apply(
            lambda x: str(x.year + 1) if x.month >= 4 else str(x.year),
        )
        print(f"kpi_val_test size: {len(kpi_val_test)}")

        kpi_val_test["pred_target_clip"] = kpi_val_test[self.pred_target_name].clip(
            -0.5,
            0.5,
        )
        kpi_val_test_pivot = pd.pivot_table(
            kpi_val_test.query("fiscal_year in ['2021','2022','2023','2024','2025']"),
            index="response_edinetCode",
            columns="fiscal_year",
            values="pred_target_clip",
            aggfunc=np.mean,
        ).fillna(0)

        # train企業（retrieval pool）のKPI推移
        kpi_val_retrieval = kpi_val.query("index in @self.retrieval_pool_kpi_5.index")
        kpi_val_retrieval["response_edinetCode"] = self.retrieval_pool_kpi_5[
            "response_edinetCode"
        ]
        kpi_val_retrieval["period_end_dt"] = self.retrieval_pool_kpi_5["period_end_dt"]

        # 会計年度を計算（3/31ベース: 2020/4-2021/3 -> '2021'）
        # period_end_dt（決算日）が2021/3/31なら'2021'年度、2021/4/1以降なら'2022'年度
        kpi_val_retrieval["fiscal_year"] = kpi_val_retrieval["period_end_dt"].apply(
            lambda x: str(x.year + 1) if x.month >= 4 else str(x.year),
        )
        print(f"kpi_val_retrieval size: {len(kpi_val_retrieval)}")

        kpi_val_retrieval["pred_target_clip"] = kpi_val_retrieval[
            self.pred_target_name
        ].clip(
            -0.5,
            0.5,
        )
        kpi_val_retrieval_pivot = pd.pivot_table(
            kpi_val_retrieval.query(
                "fiscal_year in ['2021','2022','2023','2024','2025']",
            ),
            index="response_edinetCode",
            columns="fiscal_year",
            values="pred_target_clip",
            aggfunc=np.mean,
        ).fillna(0)

        if filter_by_quarter:
            # クォーター情報を追加
            kpi_val_test["fiscal_quarter"] = kpi_val_test["period_end_dt"].apply(
                get_fiscal_quarter,
            )
            kpi_val_retrieval["fiscal_quarter"] = kpi_val_retrieval[
                "period_end_dt"
            ].apply(
                get_fiscal_quarter,
            )

            # edinetcodeごとのクォーターマッピングを作成
            test_quarter_map = (
                kpi_val_test.groupby("response_edinetCode")["fiscal_quarter"]
                .first()
                .to_dict()
            )
            retrieval_quarter_map = (
                kpi_val_retrieval.groupby("response_edinetCode")["fiscal_quarter"]
                .first()
                .to_dict()
            )

            print(
                f"Test quarters distribution: {kpi_val_test['fiscal_quarter'].value_counts().to_dict()}",
            )
            print(
                f"Retrieval quarters distribution: {kpi_val_retrieval['fiscal_quarter'].value_counts().to_dict()}",
            )

        # 正解リストを作成
        ans_dict = {}
        for edinet_code in kpi_val_test_pivot.index:
            if filter_by_quarter:
                # test企業のクォーターを取得
                test_quarter = test_quarter_map.get(edinet_code)
                if test_quarter is None:
                    continue

                # 同じクォーターのretrieval企業のみをフィルタリング
                same_quarter_retrieval_codes = [
                    code
                    for code, quarter in retrieval_quarter_map.items()
                    if quarter == test_quarter and code in kpi_val_retrieval_pivot.index
                ]

                if len(same_quarter_retrieval_codes) == 0:
                    continue

                # 同じクォーターのretrieval企業のみでpivotテーブルを作成
                kpi_val_retrieval_pivot_filtered = kpi_val_retrieval_pivot.loc[
                    same_quarter_retrieval_codes
                ]
            else:
                # クォーターフィルタなし：全retrieval企業を対象
                kpi_val_retrieval_pivot_filtered = kpi_val_retrieval_pivot

            similar_companies = self._get_similar_companies_cross_pool(
                edinet_code,
                kpi_val_test_pivot,
                kpi_val_retrieval_pivot_filtered,
                top_k=ans_top_k,
            )
            if similar_companies is not None:
                ans_dict[edinet_code] = set(similar_companies.index)

        print(f"ans_dict size: {len(ans_dict)}")
        print(f"ans_dict sample (first 3): {list(ans_dict.keys())[:3]}")

        # 予測リストを作成
        if filter_by_quarter:
            pred_ranked_lists_dict = {}
            for source_edinet in distance_df["source_edinetcode"].unique():
                # source企業のクォーターを取得
                source_quarter = test_quarter_map.get(source_edinet)
                if source_quarter is None:
                    continue

                # 同じクォーターのtarget企業のみをフィルタリング
                same_quarter_targets = [
                    code
                    for code, quarter in retrieval_quarter_map.items()
                    if quarter == source_quarter
                ]

                # distance_dfから同じクォーターの組み合わせのみを抽出
                filtered_df = distance_df[
                    (distance_df["source_edinetcode"] == source_edinet)
                    & (distance_df["target_edinetcode"].isin(same_quarter_targets))
                ]

                if len(filtered_df) > 0:
                    sorted_df = filtered_df.sort_values(by=col_name)
                    pred_ranked_lists_dict[source_edinet] = sorted_df[
                        "target_edinetcode"
                    ].to_list()

            pred_ranked_lists = pd.Series(pred_ranked_lists_dict)
        else:
            # クォーターフィルタなし：通常の距離順ランキング
            pred_ranked_lists = distance_df.groupby("source_edinetcode").apply(
                self._get_ranked_list,
                col_name=col_name,
            )

        print(f"pred_ranked_lists size: {len(pred_ranked_lists)}")
        print(
            f"pred_ranked_lists sample (first 3): {list(pred_ranked_lists.index)[:3]}",
        )

        # オーバーラップを確認
        ans_keys = set(ans_dict.keys())
        pred_keys = set(pred_ranked_lists.index)
        overlap = ans_keys & pred_keys
        print(f"ans_dict keys count: {len(ans_keys)}")
        print(f"pred_ranked_lists keys count: {len(pred_keys)}")
        print(f"overlap count: {len(overlap)}")

        # MAP@kを計算
        ap_scores = []
        for edinet_code, relevant_set in ans_dict.items():
            if edinet_code not in pred_ranked_lists.index:
                continue
            predicted_list = pred_ranked_lists[edinet_code]
            ap = self._average_precision_at_k(predicted_list, relevant_set, ans_top_k)
            ap_scores.append(ap)

        map_score = np.mean(ap_scores) if ap_scores else 0.0
        if len(ap_scores) == 0:
            print(f"ap_scores is empty: {ap_scores}")
        else:
            print(f"ap_scores count: {len(ap_scores)}, mean: {map_score}")
        return map_score

    def eval_distance_kpi_trend_mrr(
        self,
        distance_df,
        col_name="ot_distance",
        ans_top_k=5,
        filter_by_quarter=False,
    ):
        """正解数kを与えた時のMean Reciprocal Rank (MRR)を計算する

        k個の正解がある場合、最初にヒットした正解の順位の逆数を計算し、
        全クエリで平均を取る。

        Parameters
        ----------
        distance_df : pd.DataFrame
            distance_df.columns: ["source_edinetcode", "target_edinetcode", col_name]
        col_name : str
            距離のカラム名
        ans_top_k : int
            正解数k (KPI年次推移でユークリッド距離が近いトップk社を正解とする)
        filter_by_quarter : bool, default False
            Trueの場合、決算期（クォーター）が同じ企業同士でのみ検索を行う

        Returns
        -------
        float
            MRR score

        """
        kpi_val = load_kpi_val()

        # test企業のKPI推移
        kpi_val_test = kpi_val.query("index in @self.eval_kpi_test_5.index")
        kpi_val_test["response_edinetCode"] = self.eval_kpi_test_5[
            "response_edinetCode"
        ]
        kpi_val_test["period_end_dt"] = self.eval_kpi_test_5["period_end_dt"]

        # 会計年度を計算（3/31ベース: 2020/4-2021/3 -> '2021'）
        # period_end_dt（決算日）が2021/3/31なら'2021'年度、2021/4/1以降なら'2022'年度
        kpi_val_test["fiscal_year"] = kpi_val_test["period_end_dt"].apply(
            lambda x: str(x.year + 1) if x.month >= 4 else str(x.year),
        )
        print(f"[MRR] kpi_val_test size: {len(kpi_val_test)}")

        kpi_val_test["pred_target_clip"] = kpi_val_test[self.pred_target_name].clip(
            -0.5,
            0.5,
        )

        # eval_kpi_test_5の会計年度も計算
        eval_kpi_test_5_fiscal = self.eval_kpi_test_5.copy()
        eval_kpi_test_5_fiscal["fiscal_year"] = eval_kpi_test_5_fiscal[
            "period_end_dt"
        ].apply(
            lambda x: str(x.year + 1) if x.month >= 4 else str(x.year),
        )

        kpi_val_test_pivot = pd.pivot_table(
            kpi_val_test.query(
                "fiscal_year in @eval_kpi_test_5_fiscal.fiscal_year.unique()",
            ),
            index="response_edinetCode",
            columns="fiscal_year",
            values="pred_target_clip",
            aggfunc=np.mean,
        ).fillna(0)

        # train企業（retrieval pool）のKPI推移
        kpi_val_retrieval = kpi_val.query("index in @self.retrieval_pool_kpi_5.index")
        kpi_val_retrieval["response_edinetCode"] = self.retrieval_pool_kpi_5[
            "response_edinetCode"
        ]
        kpi_val_retrieval["period_end_dt"] = self.retrieval_pool_kpi_5["period_end_dt"]

        # 会計年度を計算（3/31ベース: 2020/4-2021/3 -> '2021'）
        # period_end_dt（決算日）が2021/3/31なら'2021'年度、2021/4/1以降なら'2022'年度
        kpi_val_retrieval["fiscal_year"] = kpi_val_retrieval["period_end_dt"].apply(
            lambda x: str(x.year + 1) if x.month >= 4 else str(x.year),
        )
        print(f"[MRR] kpi_val_retrieval size: {len(kpi_val_retrieval)}")

        kpi_val_retrieval["pred_target_clip"] = kpi_val_retrieval[
            self.pred_target_name
        ].clip(
            -0.5,
            0.5,
        )
        kpi_val_retrieval_pivot = pd.pivot_table(
            kpi_val_retrieval.query(
                "fiscal_year in ['2021','2022','2023','2024','2025']",
            ),
            index="response_edinetCode",
            columns="fiscal_year",
            values="pred_target_clip",
            aggfunc=np.mean,
        ).fillna(0)

        if filter_by_quarter:
            # クォーター情報を追加
            kpi_val_test["fiscal_quarter"] = kpi_val_test["period_end_dt"].apply(
                get_fiscal_quarter,
            )
            kpi_val_retrieval["fiscal_quarter"] = kpi_val_retrieval[
                "period_end_dt"
            ].apply(
                get_fiscal_quarter,
            )

            # edinetcodeごとのクォーターマッピングを作成
            test_quarter_map = (
                kpi_val_test.groupby("response_edinetCode")["fiscal_quarter"]
                .first()
                .to_dict()
            )
            retrieval_quarter_map = (
                kpi_val_retrieval.groupby("response_edinetCode")["fiscal_quarter"]
                .first()
                .to_dict()
            )

            print(
                f"[MRR] Test quarters distribution: {kpi_val_test['fiscal_quarter'].value_counts().to_dict()}",
            )
            print(
                f"[MRR] Retrieval quarters distribution: {kpi_val_retrieval['fiscal_quarter'].value_counts().to_dict()}",
            )

        # 正解リストを作成
        ans_dict = {}
        for edinet_code in kpi_val_test_pivot.index:
            if filter_by_quarter:
                # test企業のクォーターを取得
                test_quarter = test_quarter_map.get(edinet_code)
                if test_quarter is None:
                    continue

                # 同じクォーターのretrieval企業のみをフィルタリング
                same_quarter_retrieval_codes = [
                    code
                    for code, quarter in retrieval_quarter_map.items()
                    if quarter == test_quarter and code in kpi_val_retrieval_pivot.index
                ]

                if len(same_quarter_retrieval_codes) == 0:
                    continue

                # 同じクォーターのretrieval企業のみでpivotテーブルを作成
                kpi_val_retrieval_pivot_filtered = kpi_val_retrieval_pivot.loc[
                    same_quarter_retrieval_codes
                ]
            else:
                # クォーターフィルタなし：全retrieval企業を対象
                kpi_val_retrieval_pivot_filtered = kpi_val_retrieval_pivot

            similar_companies = self._get_similar_companies_cross_pool(
                edinet_code,
                kpi_val_test_pivot,
                kpi_val_retrieval_pivot_filtered,
                top_k=ans_top_k,
            )
            if similar_companies is not None:
                ans_dict[edinet_code] = set(similar_companies.index)

        print(f"[MRR] ans_dict size: {len(ans_dict)}")
        print(f"[MRR] ans_dict sample (first 3): {list(ans_dict.keys())[:3]}")

        # 予測リストを作成
        if filter_by_quarter:
            pred_ranked_lists_dict = {}
            for source_edinet in distance_df["source_edinetcode"].unique():
                # source企業のクォーターを取得
                source_quarter = test_quarter_map.get(source_edinet)
                if source_quarter is None:
                    continue

                # 同じクォーターのtarget企業のみをフィルタリング
                same_quarter_targets = [
                    code
                    for code, quarter in retrieval_quarter_map.items()
                    if quarter == source_quarter
                ]

                # distance_dfから同じクォーターの組み合わせのみを抽出
                filtered_df = distance_df[
                    (distance_df["source_edinetcode"] == source_edinet)
                    & (distance_df["target_edinetcode"].isin(same_quarter_targets))
                ]

                if len(filtered_df) > 0:
                    sorted_df = filtered_df.sort_values(by=col_name)
                    pred_ranked_lists_dict[source_edinet] = sorted_df[
                        "target_edinetcode"
                    ].to_list()

            pred_ranked_lists = pd.Series(pred_ranked_lists_dict)
        else:
            # クォーターフィルタなし：通常の距離順ランキング
            pred_ranked_lists = distance_df.groupby("source_edinetcode").apply(
                self._get_ranked_list,
                col_name=col_name,
            )

        print(f"[MRR] pred_ranked_lists size: {len(pred_ranked_lists)}")
        print(
            f"[MRR] pred_ranked_lists sample (first 3): {list(pred_ranked_lists.index)[:3]}",
        )

        # オーバーラップを確認
        ans_keys = set(ans_dict.keys())
        pred_keys = set(pred_ranked_lists.index)
        overlap = ans_keys & pred_keys
        print(f"[MRR] ans_dict keys count: {len(ans_keys)}")
        print(f"[MRR] pred_ranked_lists keys count: {len(pred_keys)}")
        print(f"[MRR] overlap count: {len(overlap)}")

        # MRRを計算
        rr_scores = []
        for edinet_code, relevant_set in ans_dict.items():
            if edinet_code not in pred_ranked_lists.index:
                continue
            predicted_list = pred_ranked_lists[edinet_code]
            rr = self._reciprocal_rank(predicted_list, relevant_set)
            rr_scores.append(rr)

        mrr_score = np.mean(rr_scores) if rr_scores else 0.0
        if len(rr_scores) == 0:
            print(f"[MRR] rr_scores is empty: {rr_scores}")
        else:
            print(f"[MRR] rr_scores count: {len(rr_scores)}, mean: {mrr_score}")
        return mrr_score

    def _get_ranked_list(self, sr, col_name="ot_distance"):
        """距離順でソートした企業リストを取得する。"""
        sr = sr.sort_values(by=col_name)
        return sr.target_edinetcode.to_list()

    def _get_similar_companies_cross_pool(
        self,
        target_edinet_code,
        source_pivot,
        target_pivot,
        top_k=5,
    ):
        """source企業のKPI推移に類似したtarget企業トップKを取得する

        Parameters
        ----------
        target_edinet_code : str
            対象企業のEDINETコード（source_pivotから）
        source_pivot : pd.DataFrame
            source企業のKPI推移pivot table
        target_pivot : pd.DataFrame
            target企業（検索対象）のKPI推移pivot table
        top_k : int
            取得する類似企業数

        Returns
        -------
        pd.Series
            ユークリッド距離が小さい順のtarget企業リスト

        """
        from sklearn.metrics.pairwise import euclidean_distances

        if target_edinet_code not in source_pivot.index:
            return None

        # 共通の年のカラムのみを使用
        common_cols = list(set(source_pivot.columns) & set(target_pivot.columns))
        if len(common_cols) == 0:
            return None

        source_trend = source_pivot.loc[target_edinet_code, common_cols].values.reshape(
            1,
            -1,
        )
        target_trends = target_pivot[common_cols].values

        # ユークリッド距離を計算
        distances = euclidean_distances(source_trend, target_trends)[0]

        # 距離の小さい順にソート
        top_indices = np.argsort(distances)[:top_k]
        result = pd.Series(
            distances[top_indices],
            index=target_pivot.index[top_indices],
        )

        return result

    def _reciprocal_rank(self, predicted_list, relevant_set):
        """Reciprocal Rankを計算する

        最初に正解が出現した順位の逆数を返す。

        Parameters
        ----------
        predicted_list : list
            予測された順序付きリスト (距離が近い順)
        relevant_set : set
            正解アイテムのセット

        Returns
        -------
        float
            Reciprocal Rank (1/rank)。正解が見つからない場合は0.0

        """
        for i, item in enumerate(predicted_list):
            if item in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def _average_precision_at_k(self, predicted_list, relevant_set, k):
        """Average Precision@kを計算する

        Parameters
        ----------
        predicted_list : list
            予測された順序付きリスト（距離が近い順）
        relevant_set : set
            正解アイテムのセット
        k : int
            正解数

        Returns
        -------
        float
            AP@k score

        """
        if len(relevant_set) == 0:
            return 0.0

        score = 0.0
        num_hits = 0

        for i, item in enumerate(predicted_list):
            if item in relevant_set:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                score += precision_at_i

        # 正解数kで割る（正解が全て見つからなくても最大k個の正解があるとして正規化）
        return score / k

    # ------------------------------------------------------------------
    # 相関係数ベース評価指標 (Kendall tau / nDCG / Recall@K)
    # ------------------------------------------------------------------

    def _build_kpi_pivots(self):
        """test企業・retrieval企業それぞれのKPI推移pivot tableを構築する。"""
        kpi_val = load_kpi_val()

        kpi_val_test = kpi_val.query("index in @self.eval_kpi_test_5.index")
        kpi_val_test["response_edinetCode"] = self.eval_kpi_test_5["response_edinetCode"]
        kpi_val_test["period_end_dt"] = self.eval_kpi_test_5["period_end_dt"]
        kpi_val_test["fiscal_year"] = kpi_val_test["period_end_dt"].apply(
            lambda x: str(x.year + 1) if x.month >= 4 else str(x.year),
        )
        kpi_val_test["pred_target_clip"] = kpi_val_test[self.pred_target_name].clip(-0.5, 0.5)
        test_pivot = pd.pivot_table(
            kpi_val_test.query("fiscal_year in ['2021','2022','2023','2024','2025']"),
            index="response_edinetCode",
            columns="fiscal_year",
            values="pred_target_clip",
            aggfunc=np.mean,
        ).fillna(0)

        kpi_val_retrieval = kpi_val.query("index in @self.retrieval_pool_kpi_5.index")
        kpi_val_retrieval["response_edinetCode"] = self.retrieval_pool_kpi_5["response_edinetCode"]
        kpi_val_retrieval["period_end_dt"] = self.retrieval_pool_kpi_5["period_end_dt"]
        kpi_val_retrieval["fiscal_year"] = kpi_val_retrieval["period_end_dt"].apply(
            lambda x: str(x.year + 1) if x.month >= 4 else str(x.year),
        )
        kpi_val_retrieval["pred_target_clip"] = kpi_val_retrieval[self.pred_target_name].clip(
            -0.5, 0.5,
        )
        retrieval_pivot = pd.pivot_table(
            kpi_val_retrieval.query("fiscal_year in ['2021','2022','2023','2024','2025']"),
            index="response_edinetCode",
            columns="fiscal_year",
            values="pred_target_clip",
            aggfunc=np.mean,
        ).fillna(0)

        print(
            f"[pivot] test: {test_pivot.shape}, retrieval: {retrieval_pivot.shape}",
        )
        return test_pivot, retrieval_pivot

    def _compute_kpi_correlation_matrix(self, source_pivot, target_pivot):
        """source企業×target企業のPearson相関係数行列を行列演算で一括計算する。

        Returns
        -------
        pd.DataFrame
            shape (n_source, n_target), index=source_pivot.index, columns=target_pivot.index
            値域 [-1, 1]、高いほど類似
        """
        common_cols = sorted(set(source_pivot.columns) & set(target_pivot.columns))
        if len(common_cols) < 2:
            return None

        A = source_pivot[common_cols].values.astype(float)  # (n_source, n_years)
        B = target_pivot[common_cols].values.astype(float)  # (n_target, n_years)

        A_c = A - A.mean(axis=1, keepdims=True)
        B_c = B - B.mean(axis=1, keepdims=True)

        A_norm = np.linalg.norm(A_c, axis=1, keepdims=True)
        B_norm = np.linalg.norm(B_c, axis=1, keepdims=True)

        # ゼロ分散行をゼロベクトルに
        A_norm = np.where(A_norm < 1e-10, 1.0, A_norm)
        B_norm = np.where(B_norm < 1e-10, 1.0, B_norm)

        A_u = A_c / A_norm
        B_u = B_c / B_norm

        corr_matrix = np.clip(A_u @ B_u.T, -1.0, 1.0)  # (n_source, n_target)

        return pd.DataFrame(
            corr_matrix,
            index=source_pivot.index,
            columns=target_pivot.index,
        )

    def _compute_kpi_distance_matrix(self, source_pivot, target_pivot, metric="l1"):
        """source企業×target企業のL1またはL2距離行列を行列演算で一括計算する。

        Parameters
        ----------
        metric : str
            "l1" (Manhattan距離) または "l2" (Euclidean距離)

        Returns
        -------
        pd.DataFrame
            shape (n_source, n_target), index=source_pivot.index, columns=target_pivot.index
            値域 [0, +inf)、小さいほど類似
        """
        common_cols = sorted(set(source_pivot.columns) & set(target_pivot.columns))
        if len(common_cols) < 1:
            return None

        from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

        A = source_pivot[common_cols].values.astype(float)
        B = target_pivot[common_cols].values.astype(float)

        if metric == "l1":
            dist_matrix = manhattan_distances(A, B)
        elif metric == "l2":
            dist_matrix = euclidean_distances(A, B)
        else:
            msg = f"Unknown metric: {metric}. Use 'l1' or 'l2'."
            raise ValueError(msg)

        return pd.DataFrame(
            dist_matrix,
            index=source_pivot.index,
            columns=target_pivot.index,
        )

    def eval_distance_kpi_trend_kendalltau(
        self,
        distance_df,
        col_name="dist",
        gt_metric="correlation",
    ):
        """Kendall tau: モデルランキング vs KPIトレンド類似度ランキングの順位整合性。

        各クエリ企業について、モデルが予測したtarget企業の順位と
        KPIトレンドの類似度（correlation/l1/l2）に基づく順位のKendall tau-bを計算し平均する。

        Parameters
        ----------
        distance_df : pd.DataFrame
            ["source_edinetcode", "target_edinetcode", col_name]
        col_name : str
            距離カラム名（小さいほど近い）
        gt_metric : str
            正解ランキングの算出方法。"correlation" (Pearson相関係数、高いほど類似),
            "l1" (Manhattan距離、小さいほど類似), "l2" (Euclidean距離、小さいほど類似)

        Returns
        -------
        float
            mean Kendall tau-b
        """
        from scipy.stats import kendalltau

        test_pivot, retrieval_pivot = self._build_kpi_pivots()

        if gt_metric == "correlation":
            score_matrix = self._compute_kpi_correlation_matrix(test_pivot, retrieval_pivot)
            higher_is_better = True
        else:
            score_matrix = self._compute_kpi_distance_matrix(
                test_pivot, retrieval_pivot, metric=gt_metric,
            )
            higher_is_better = False

        if score_matrix is None:
            return 0.0

        pred_ranked_lists = distance_df.groupby("source_edinetcode").apply(
            self._get_ranked_list,
            col_name=col_name,
        )

        tau_scores = []
        for source_code in score_matrix.index:
            if source_code not in pred_ranked_lists.index:
                continue

            predicted_list = [
                t for t in pred_ranked_lists[source_code] if t in score_matrix.columns
            ]
            if len(predicted_list) < 3:
                continue

            # モデルの予測順位（0-indexed）
            model_rank_arr = np.arange(len(predicted_list), dtype=float)
            scores = score_matrix.loc[source_code, predicted_list].values
            # 類似度は降順、距離は昇順でランク付け
            if higher_is_better:
                gt_rank_arr = (-scores).argsort().argsort().astype(float)
            else:
                gt_rank_arr = scores.argsort().argsort().astype(float)

            tau_val, _ = kendalltau(model_rank_arr, gt_rank_arr)
            if not np.isnan(tau_val):
                tau_scores.append(tau_val)

        mean_score = float(np.mean(tau_scores)) if tau_scores else 0.0
        print(
            f"[KendallTau({gt_metric})] n_queries={len(tau_scores)}, mean={mean_score:.4f}",
        )
        return mean_score

    def eval_distance_kpi_trend_ndcg(
        self,
        distance_df,
        col_name="dist",
        k=32,
        gt_metric="correlation",
    ):
        """nDCG@K: KPIトレンド類似度を関連度スコアとして使用したnDCGを計算する。

        correlation: 関連度 = max(0, Pearson相関係数)
        l1/l2:       関連度 = 1 / (1 + distance) — 常に正、距離が小さいほど高い

        Parameters
        ----------
        distance_df : pd.DataFrame
            ["source_edinetcode", "target_edinetcode", col_name]
        col_name : str
            距離カラム名
        k : int
            評価するランキング上位K
        gt_metric : str
            正解類似度の算出方法。"correlation", "l1", "l2"

        Returns
        -------
        float
            mean nDCG@K
        """
        test_pivot, retrieval_pivot = self._build_kpi_pivots()

        if gt_metric == "correlation":
            score_matrix = self._compute_kpi_correlation_matrix(test_pivot, retrieval_pivot)
            def _to_relevance(score):
                return max(0.0, score)
        else:
            score_matrix = self._compute_kpi_distance_matrix(
                test_pivot, retrieval_pivot, metric=gt_metric,
            )
            def _to_relevance(score):
                return 1.0 / (1.0 + score)  # score=距離、小さいほど高い関連度

        if score_matrix is None:
            return 0.0

        pred_ranked_lists = distance_df.groupby("source_edinetcode").apply(
            self._get_ranked_list,
            col_name=col_name,
        )

        ndcg_scores = []
        for source_code in score_matrix.index:
            if source_code not in pred_ranked_lists.index:
                continue

            predicted_top_k = [
                t for t in pred_ranked_lists[source_code]
                if t in score_matrix.columns
            ][:k]

            dcg = sum(
                _to_relevance(score_matrix.loc[source_code, t]) / np.log2(rank + 2)
                for rank, t in enumerate(predicted_top_k)
            )

            all_rels = np.array([
                _to_relevance(v) for v in score_matrix.loc[source_code].values
            ])
            ideal_rels = np.sort(all_rels)[::-1][:k]
            idcg = sum(
                rel / np.log2(rank + 2)
                for rank, rel in enumerate(ideal_rels)
                if rel > 0
            )

            if idcg == 0:
                continue

            ndcg_scores.append(dcg / idcg)

        mean_score = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
        print(f"[nDCG@{k}({gt_metric})] n_queries={len(ndcg_scores)}, mean={mean_score:.4f}")
        return mean_score

    def eval_distance_kpi_trend_recall_at_k(
        self,
        distance_df,
        col_name="dist",
        k=32,
        threshold=0.5,
        gt_metric="correlation",
    ):
        """Recall@K（取りこぼし指標）: 閾値で関連企業を定義し上位K社の取りこぼしを評価。

        gt_metric="correlation": KPI Pearson相関係数 >= threshold の企業を「関連あり」とする。
        gt_metric="l1"/"l2":    KPI L1/L2距離 <= threshold の企業を「関連あり」とする。

        Parameters
        ----------
        distance_df : pd.DataFrame
            ["source_edinetcode", "target_edinetcode", col_name]
        col_name : str
            距離カラム名
        k : int
            上位K社
        threshold : float
            関連企業の閾値。
            correlation: 相関係数の下限（default 0.5, 値域 [-1, 1]）
            l1: L1距離の上限（KPIをclip[-0.5, 0.5]×5年の場合、最大距離 ~5.0）
            l2: L2距離の上限（最大距離 ~2.2）
        gt_metric : str
            正解類似度の算出方法。"correlation", "l1", "l2"

        Returns
        -------
        float
            mean Recall@K
        """
        test_pivot, retrieval_pivot = self._build_kpi_pivots()

        if gt_metric == "correlation":
            score_matrix = self._compute_kpi_correlation_matrix(test_pivot, retrieval_pivot)
            def _is_relevant(scores):
                return scores >= threshold
        else:
            score_matrix = self._compute_kpi_distance_matrix(
                test_pivot, retrieval_pivot, metric=gt_metric,
            )
            def _is_relevant(scores):
                return scores <= threshold  # 距離なので上限で絞る

        if score_matrix is None:
            return 0.0

        pred_ranked_lists = distance_df.groupby("source_edinetcode").apply(
            self._get_ranked_list,
            col_name=col_name,
        )

        recall_scores = []
        n_relevant_list = []
        for source_code in score_matrix.index:
            if source_code not in pred_ranked_lists.index:
                continue

            relevant_targets = set(
                score_matrix.columns[_is_relevant(score_matrix.loc[source_code])],
            )
            if len(relevant_targets) == 0:
                continue

            predicted_top_k = {
                t for t in pred_ranked_lists[source_code][:k]
                if t in score_matrix.columns
            }
            recall = len(relevant_targets & predicted_top_k) / len(relevant_targets)
            recall_scores.append(recall)
            n_relevant_list.append(len(relevant_targets))

        mean_score = float(np.mean(recall_scores)) if recall_scores else 0.0
        avg_relevant = float(np.mean(n_relevant_list)) if n_relevant_list else 0.0
        print(
            f"[Recall@{k}({gt_metric},thr={threshold})] "
            f"n_queries={len(recall_scores)}, mean={mean_score:.4f}, "
            f"avg_relevant={avg_relevant:.1f}",
        )
        return mean_score


# %%


# %%


def get_edineccode_dist_pred_kpi(distance_df, col_name="dist"):
    """Add edinet_code and year to the source and target of the distance_df
    to calc for kpi prediction per edinet_code.

    距離計算はtrain termの最終年のdocidで行われる（リーク防止）が、
    評価時にはeval termで5年分データがある企業のみを対象とする。

    手順:
    1. train termの最終年のdocidでフィルタリング
    2. edinetcodeに変換
    3. eval_kpi_test_5に含まれる企業のみに絞る

    """
    (
        eval_kpi_test_df,
        retrieval_pool_kpi_df,
        eval_kpi_test_5,
        retrieval_pool_kpi_5,
    ) = get_task_div_pred_kpi(
        eval_periodend_startdate=cfg["eval_periodend_startdate"],
        eval_periodend_enddate=cfg["eval_periodend_enddate"],
        train_periodend_startdate=cfg["train_periodend_startdate"],
        train_periodend_enddate=cfg["train_periodend_enddate"],
    )  # kpi pred

    # train termの最終年のdocidでフィルタリング（リーク防止）
    # source: test企業, target: train企業
    distance_df = distance_df.query(
        "source_docid in @eval_kpi_test_df.index and target_docid in @retrieval_pool_kpi_df.index",
    )
    print("distance_calc_shape (train term docid):", distance_df.shape)

    # edinetcodeに変換
    clip_distance_sector2 = distance_df.merge(
        eval_kpi_test_df[["response_edinetCode", "year"]].rename(
            columns={"response_edinetCode": "source_edinetcode", "year": "source_year"},
        ),
        left_on="source_docid",
        right_index=True,
        how="left",
    ).merge(
        retrieval_pool_kpi_df[["response_edinetCode", "year"]].rename(
            columns={"response_edinetCode": "target_edinetcode", "year": "target_year"},
        ),
        left_on="target_docid",
        right_index=True,
        how="left",
    )
    print("shape after edinetcode merge:", clip_distance_sector2.shape)

    # sourceはeval_kpi_test_5、targetはretrieval_pool_kpi_5に含まれる企業のみに絞る
    eval_edinetcode_set = set(eval_kpi_test_5["response_edinetCode"].unique())
    retrieval_edinetcode_set = set(retrieval_pool_kpi_5["response_edinetCode"].unique())
    print(f"eval_edinetcode_set size: {len(eval_edinetcode_set)}")
    print(f"retrieval_edinetcode_set size: {len(retrieval_edinetcode_set)}")

    print(
        f"unique source_edinetcode before filter: {clip_distance_sector2['source_edinetcode'].nunique()}",
    )
    print(
        f"unique target_edinetcode before filter: {clip_distance_sector2['target_edinetcode'].nunique()}",
    )

    clip_distance_sector2 = clip_distance_sector2.query(
        "source_edinetcode in @eval_edinetcode_set and target_edinetcode in @retrieval_edinetcode_set",
    )
    print(
        "shape after filtering:",
        clip_distance_sector2.shape,
    )
    print(
        f"unique source_edinetcode after filter: {clip_distance_sector2['source_edinetcode'].nunique()}",
    )
    print(
        f"unique target_edinetcode after filter: {clip_distance_sector2['target_edinetcode'].nunique()}",
    )
    # clip_distance_edinetcode = (
    #    clip_distance_sector2.query(
    #        "source_edinetcode.notna() and target_edinetcode.notna() and source_year == '2020' and target_year == '2020'",
    #    )
    #    .groupby(
    #        ["source_edinetcode", "target_edinetcode"],
    #    )
    #    .agg({col_name: "mean"})
    # )
    return clip_distance_sector2


# %% ##################################################################
#
#                            Results
#
########################################################################
def get_results_kpi():
    # filename = DATADIR / "response_tbl_dataset_train_260218.pkl"
    # response_tbl_train = pd.read_pickle(filename)
    # response_tbl_all = get_all_response_tbl()
    # (
    #    eval_response_df,
    #    retrieval_pool_df,
    #    eval_response_5_df,
    #    retrieval_pool_response_5_df,
    # ) = get_task_div_pred_kpi(
    #    eval_periodend_startdate="2020-04-01",
    #    eval_periodend_enddate="2025-03-31",
    #    train_periodend_startdate="2014-01-01",
    #    train_periodend_enddate="2020-03-31",
    # )  # kpi covariate pred

    print("load_ot_distance_list_kpi")

    calc_list = []
    for method_name in cfg["method_name_list"]:
        distance_kpi = load_distance_kpi(method=method_name)
        distance_kpi["dist"] = 1 - distance_kpi["dist"]
        distance_kpi_edinetcode = get_edineccode_dist_pred_kpi(
            distance_kpi,
        )
        calc_list.append(
            {
                "method_name": method_name,
                "distance_kpi_edinetcode": distance_kpi_edinetcode,
            },
        )
    # correlation
    print("load_cor_distance_kpi")
    cor_distance_kpi = load_cor_distance_kpi()
    cor_distance_kpi_edinetcode = get_edineccode_dist_pred_kpi(cor_distance_kpi)
    calc_list.append(
        {
            "method_name": "correlation",
            "distance_kpi_edinetcode": cor_distance_kpi_edinetcode,
        },
    )
    # mahalanobis
    mahalanobis_distance_kpi = load_mahalanobis_distance_kpi()
    mahalanobis_distance_kpi_edinetcode = get_edineccode_dist_pred_kpi(
        mahalanobis_distance_kpi,
    )
    calc_list.append(
        {
            "method_name": "mahalanobis",
            "distance_kpi_edinetcode": mahalanobis_distance_kpi_edinetcode,
        },
    )
    # bm25
    distance_kpi = load_distance_kpi(method="bm25")  # already transformed to distance
    distance_kpi_edinetcode = get_edineccode_dist_pred_kpi(
        distance_kpi,
    )
    calc_list.append(
        {
            "method_name": "bm25",
            "distance_kpi_edinetcode": distance_kpi_edinetcode,
        },
    )
    results = []
    # Kendall tau順位相関（3指標）
    #for gt_metric in ["correlation"]:
    #    for calc in calc_list:
    #        for pred_target_name in ["kpi_diff_rate", "revinue_diff_rate"]:
    #            results.append(
    #                {
    #                    "method": calc["method_name"],
    #                    "task": f"kendalltau_{gt_metric}_{pred_target_name}",
    #                    "score": EvalDistance(
    #                        pred_target_name=pred_target_name,
    #                    ).eval_distance_kpi_trend_kendalltau(
    #                        calc["distance_kpi_edinetcode"].reset_index(),
    #                        col_name="dist",
    #                        gt_metric=gt_metric,
    #                    ),
    #                },
    #            )
    # nDCG@K（3指標 × 2K）
    for gt_metric in ["correlation"]:
        for k in cfg["thr_kpi_ndcg_list"]:
            for calc in calc_list:
                for pred_target_name in ["kpi_diff_rate", "revinue_diff_rate"]:
                    results.append(
                        {
                            "method": calc["method_name"],
                            "task": f"ndcg_k{k}_{gt_metric}_{pred_target_name}",
                            "score": EvalDistance(
                                pred_target_name=pred_target_name,
                            ).eval_distance_kpi_trend_ndcg(
                                calc["distance_kpi_edinetcode"].reset_index(),
                                col_name="dist",
                                k=k,
                                gt_metric=gt_metric,
                            ),
                        },
                    )
    # Recall@K: 相関係数閾値ベース
    for k, thr in cfg["thr_kpi_recall_list"]:
        for calc in calc_list:
            for pred_target_name in ["kpi_diff_rate", "revinue_diff_rate"]:
                results.append(
                    {
                        "method": calc["method_name"],
                        "task": f"recall_k{k}_corr{int(thr*10)}_{pred_target_name}",
                        "score": EvalDistance(
                            pred_target_name=pred_target_name,
                        ).eval_distance_kpi_trend_recall_at_k(
                            calc["distance_kpi_edinetcode"].reset_index(),
                            col_name="dist",
                            k=k,
                            threshold=thr,
                            gt_metric="correlation",
                        ),
                    },
                )
    # MAP@K
    for k in cfg["thr_kpi_map_list"]:
        for calc in calc_list:
            for pred_target_name in ["kpi_diff_rate", "revinue_diff_rate"]:
                results.append(
                    {
                        "method": calc["method_name"],
                        "task": f"map_k{k}_{pred_target_name}",
                        "score": EvalDistance(
                            pred_target_name=pred_target_name,
                        ).eval_distance_kpi_trend_map(
                            calc["distance_kpi_edinetcode"].reset_index(),
                            col_name="dist",
                            ans_top_k=k,
                        ),
                    },
                )
    ## Recall@K: L1距離閾値ベース（clip[-0.5,0.5]×5年で最大L1~5.0）
    #for k, thr in [(32, 1.0), (128, 2.0), (32, 2.0)]:
    #    for calc in calc_list:
    #        for pred_target_name in ["kpi_diff_rate", "revinue_diff_rate"]:
    #            results.append(
    #                {
    #                    "method": calc["method_name"],
    #                    "task": f"recall_k{k}_l1thr{int(thr*10)}_{pred_target_name}",
    #                    "score": EvalDistance(
    #                        pred_target_name=pred_target_name,
    #                    ).eval_distance_kpi_trend_recall_at_k(
    #                        calc["distance_kpi_edinetcode"].reset_index(),
    #                        col_name="dist",
    #                        k=k,
    #                        threshold=thr,
    #                        gt_metric="l1",
    #                    ),
    #                },
    #            )

    results_df = pd.pivot_table(
        pd.DataFrame(results),
        index="method",
        columns="task",
        values="score",
        aggfunc=np.mean,
    )
    return results_df
    # %%
def main():
    results_df = get_results_kpi()
    out_fn = RESULTSDIR / "results_tbl" / f"results_kpi_{cfg['experiment_name']}.pkl"
    results_df.to_pickle(out_fn)
    results_df#.to_clipboard()

if __name__ == "__main__":
    main()
## %%
# (
#    eval_response_df,
#    retrieval_pool_df,
#    eval_response_5_df,
#    retrieval_pool_response_5_df,
# ) = get_task_div_pred_kpi()
## %%
# eval_response_df
# clip_distance_kpi.target_docid.nunique()
#
# %%
# clip_distance_kpi = load_clip_distance_kpi()
# clip_distance_kpi["dist"] = 1 - clip_distance_kpi["dist"]
#
# clip_distance_edinetcode = get_edineccode_dist_pred_kpi(
#    clip_distance_kpi,
# )
## %%
# clip_distance_kpi.source_docid.nunique()
## %%
# len(set(eval_response_df.index) & set(clip_distance_kpi.source_docid))
## %%
# len(set(retrieval_pool_df.index) & set(clip_distance_kpi.target_docid))
## %%
# len(set(eval_response_df.index) & set(clip_distance_edinetcode.source_docid))
## %%
# len(set(retrieval_pool_df.index) & set(clip_distance_edinetcode.target_docid))
#
# %%
#(
#    eval_kpi_test_df,
#    retrieval_pool_kpi_df,
#    eval_kpi_test_5,
#    retrieval_pool_kpi_5,
#) = get_task_div_pred_kpi()
## %%
## 会計年度を確認（3/31ベース: 2020/4-2021/3 -> '2021'）
## period_end_dt（決算日）が2021/3/31なら'2021'年度、2021/4/1以降なら'2022'年度
#eval_kpi_test_5["fiscal_year"] = eval_kpi_test_5["period_end_dt"].apply(
#    lambda x: str(x.year + 1) if x.month >= 4 else str(x.year),
#)
#eval_kpi_test_5.fiscal_year.unique()
## %%
#fn = RESULTSDIR / "results_tbl" / "results_kpi_260429.pkl"
#results_df_old = pd.read_pickle(fn)
#results_df_old
# %%
# %%
# %%
# %%
# %%
