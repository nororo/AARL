import sys

sys.path.append(r"./Projects/XBRL_common_space_projection")
sys.path.append(r"./Projects/t_interpretable_fs")

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import yaml
from pandera.typing import Series

warnings.filterwarnings("ignore")

from src.libs.load_dataset import (
    ResponseTblWithYear,
    get_all_response_tbl,
    load_bs_data,
    load_pl_data,
)

CFGDIR = Path("./Projects/t_interpretable_fs/src")
with (CFGDIR / "cfg_exp_main.yaml").open(encoding="utf-8") as _cfg_f:
    cfg = yaml.load(_cfg_f, Loader=yaml.FullLoader)

XBRL_PROJDIR = Path(cfg["xbrl_proj_path"])
PROCDIR = Path(cfg["procdir_path"])
DATADIR = Path(cfg["data_path"])


# fraud
def get_task_div_pred_fraud_old(period_end_dt_start: str, period_end_dt_end: str):
    response_tbl_all = get_all_response_tbl().query(
        "period_end_dt >= @period_end_dt_start and period_end_dt < @period_end_dt_end",
    )
    print(period_end_dt_start, " pool size: ", len(response_tbl_all))
    assert len(response_tbl_all) > 0, f"no pool data for {period_end_dt_start}"
    filename = (
        XBRL_PROJDIR
        / "data/3_processed/dataset_2507/restatement/old_0131/preproc_log_with_amd_docid_fraud_correct.pkl"
    )
    amd_docid_fraud = pd.read_pickle(filename)
    amd_docid_fraud = amd_docid_fraud.query(
        "amendment_document in @response_tbl_all.index",
    )  # 重要でないものを除外
    print(period_end_dt_start, " fraud size: ", len(amd_docid_fraud))

    # 提出tsで重複削除
    filename = (
        XBRL_PROJDIR
        / "data/0_metadata/dataset_2507/response_tbl_teisei_2507_v260131.pkl"
    )
    response_tbl_teisei = pd.read_pickle(filename)
    amd_docid_fraud = amd_docid_fraud.merge(
        response_tbl_teisei[["submitDateTime"]],
        left_on="docid",
        right_index=True,
    )
    amd_docid_fraud["submitDateTime_dt"] = pd.to_datetime(
        amd_docid_fraud["submitDateTime"],
    )
    amd_docid_fraud = amd_docid_fraud.sort_values(
        by="submitDateTime_dt",
    ).drop_duplicates(
        subset=["amendment_document"],
        keep="last",
    )

    # assert len(amd_docid_fraud) > 0, f"no fraud data for {itr_year}"
    return amd_docid_fraud.docid.to_list(), amd_docid_fraud.amendment_document.to_list()


# fraud
def get_task_div_pred_fraud_new(period_end_dt_start: str, period_end_dt_end: str):
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

    # 提出tsで重複削除
    filename = (
        XBRL_PROJDIR
        / "data/3_processed/dataset_2507/restatement/response_tbl_teisei_2507_v260131_with_year.pkl"
    )
    response_tbl_teisei = pd.read_pickle(filename)
    response_tbl_teisei = response_tbl_teisei.query(
        "year != 'not_found' and year != 'no_pre_file'",
    )
    amd_docid_fraud = amd_docid_fraud.merge(
        response_tbl_teisei[["submitDateTime"]],
        left_on="docid",
        right_index=True,
    )
    amd_docid_fraud["submitDateTime_dt"] = pd.to_datetime(
        amd_docid_fraud["submitDateTime"],
    )
    amd_docid_fraud = amd_docid_fraud.sort_values(
        by="submitDateTime_dt",
    ).drop_duplicates(
        subset=["amendment_document"],
        keep="last",
    )
    # amd_docid_fraud = amd_docid_fraud.query(
    #    "docid not in ['S100X5AB', 'S100WXZE','S100WHK9', 'S100X5AD', 'S100WXZH','S100WHKH', 'S100X9AF', 'S100X5AE', 'S100WXZL', 'S100WN6U','S100XCY6', 'S100X5AG', 'S100WOTO', 'S100WN77', 'S100WHKS', 'S100X9AC','S100WXZS', 'S100X5AH']",
    # )

    # 20250731以前まで
    # filename = (
    #    XBRL_PROJDIR
    #    / "data/3_processed/dataset_2507/restatement/response_tbl_teisei_2507_v250803_with_year.pkl"
    # )
    # response_tbl_amd_20250731 = pd.read_pickle(filename)
    # response_tbl_amd_20250731 = response_tbl_amd_20250731.query(
    #    "year != 'not_found' and year != 'no_pre_file'",
    # )
    # print(len(amd_docid_fraud))
    # amd_docid_fraud = amd_docid_fraud.query(
    #    "docid in @response_tbl_amd_20250731.index",
    # )
    # print(len(amd_docid_fraud))

    df_amounts_change_amd = pd.read_csv(PROCDIR / "df_amounts_change_amd_0403.csv")
    df_amounts_change_amd_docid = df_amounts_change_amd.docid.tolist()
    amd_docid_fraud = amd_docid_fraud.query("docid in @df_amounts_change_amd_docid")

    # amd_docid_fraud = amd_docid_fraud.query(
    #    "docid in @response_tbl_eval.index",
    # )
    # print(len(amd_docid_fraud))
    # fraud対象docidに絞る（response_evalは不正企業のdocidすべてに1を付与）
    response_tbl_fraud = pd.read_pickle(
        XBRL_PROJDIR
        / "data/3_processed/dataset_2507/restatement/response_tbl_fraud.pkl",
    )
    response_tbl_fraud = response_tbl_fraud.assign(
        period_end_dt=pd.to_datetime(response_tbl_fraud.response_periodEnd),
    ).query(
        "period_end_dt >= @period_end_dt_start and period_end_dt <= @period_end_dt_end",
    )
    fraud_docid_list = response_tbl_fraud.index.tolist()

    # assert len(amd_docid_fraud) > 0, f"no fraud data for {itr_year}"

    print("fraud documents: ", len(fraud_docid_list))


    return (
        amd_docid_fraud.docid.to_list(),  # docid_corrected
        amd_docid_fraud.amendment_document.to_list(),  # docid_error
        fraud_docid_list,  # docid_error_all
    )


# restatement
def get_task_div_pred_restatement(period_end_dt_start: str, period_end_dt_end: str):
    response_tbl_all = get_all_response_tbl().query(
        "period_end_dt >= @period_end_dt_start and period_end_dt <= @period_end_dt_end",
    )
    print(period_end_dt_start, " pool size: ", len(response_tbl_all))
    assert len(response_tbl_all) > 0, f"no pool data for {period_end_dt_start}"
    filename = (
        XBRL_PROJDIR
        / "data/3_processed/dataset_2507/restatement/preproc_log_with_amd_docid.pkl"
    )
    amd_docid_fraud = pd.read_pickle(filename)
    amd_docid_fraud = amd_docid_fraud.query(
        "amendment_document in @response_tbl_all.index",
    )  # 重要でないものを除外
    print(period_end_dt_start, " fraud size: ", len(amd_docid_fraud))

    # 提出tsで重複削除
    filename = (
        XBRL_PROJDIR
        / "data/0_metadata/dataset_2507/response_tbl_teisei_2507_v260131.pkl"
    )
    response_tbl_teisei = pd.read_pickle(filename)
    amd_docid_fraud = amd_docid_fraud.merge(
        response_tbl_teisei[["submitDateTime"]],
        left_on="docid",
        right_index=True,
    )
    amd_docid_fraud["submitDateTime_dt"] = pd.to_datetime(
        amd_docid_fraud["submitDateTime"],
    )
    amd_docid_fraud = amd_docid_fraud.sort_values(
        by="submitDateTime_dt",
    ).drop_duplicates(
        subset=["amendment_document"],
        keep="last",
    )

    # test split
    filename = DATADIR / cfg["response_tbl_test"]
    response_test = pd.read_pickle(filename)

    docid_test = response_test.query("task_val_pos_flg ==1").index

    # fraudを除外
    filename = (
        XBRL_PROJDIR
        / "data/3_processed/dataset_2507/restatement/response_tbl_fraud_all.pkl"
    )
    response_tbl_fraud_all = pd.read_pickle(filename)
    fraud_docid_list = response_tbl_fraud_all.index.tolist()
    print("before fraud size: ", len(amd_docid_fraud))
    amd_docid_fraud = amd_docid_fraud.query(
        "amendment_document in @docid_test and amendment_document not in @fraud_docid_list",
    )
    print("after fraud size: ", len(amd_docid_fraud))

    # 20250731以前までの訂正報告書
    filename = (
        XBRL_PROJDIR
        / "data/3_processed/dataset_2507/restatement/response_tbl_teisei_2507_v250803_with_year.pkl"
    )
    response_tbl_amd_20250731 = pd.read_pickle(filename)
    response_tbl_amd_20250731 = response_tbl_amd_20250731.query(
        "year != 'not_found' and year != 'no_pre_file'",
    )
    print(len(amd_docid_fraud))
    amd_docid_fraud = amd_docid_fraud.query(
        "docid in @response_tbl_amd_20250731.index",
    )
    print(len(amd_docid_fraud))
    # assert len(amd_docid_fraud) > 0, f"no fraud data for {itr_year}"
    return amd_docid_fraud.docid.to_list(), amd_docid_fraud.amendment_document.to_list()


# kpi


# train_periodend_startdate = "2014-03-31"
# train_periodend_enddate = "2020-03-31"
# eval_periodend_startdate = "2020-04-01"
# eval_periodend_enddate = "2025-03-31"


def get_task_div_pred_kpi(
    eval_periodend_startdate: str,
    eval_periodend_enddate: str,
    train_periodend_startdate: str,
    train_periodend_enddate: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Eval: 138
    Retrieval: 2000
    """
    response_tbl_all: ResponseTblWithYear = get_all_response_tbl()

    filename = DATADIR / cfg["response_tbl_train"]
    response_train = pd.read_pickle(filename)

    filename = DATADIR / cfg["response_tbl_test"]
    response_test = pd.read_pickle(filename)

    # test split
    response_test_pool = list(
        set(response_test.query("task_kpi_flg == 1").response_edinetCode),
    )
    # missing filter
    fn = PROCDIR / cfg["df_amounts_change_all"]
    df_amounts_change_all = pd.read_csv(fn)
    wo_missing_docid = list(set(df_amounts_change_all.docid))

    # term fileter
    mask = (response_tbl_all.period_end_dt >= eval_periodend_startdate) & (
        response_tbl_all.period_end_dt <= eval_periodend_enddate
    )
    response_test_pool_year_num = (
        response_tbl_all.loc[mask]
        .query(
            "response_edinetCode in @response_test_pool and index in @wo_missing_docid",
        )
        .groupby("response_edinetCode")
        .agg(
            {"period_end_dt": "nunique"},
        )
    )
    # 2021/3(2020/4/1-),2022/3,2023/3,2024/3,2025/3 (5 years)
    eval_edinet_code_list_5 = response_test_pool_year_num.query(
        "period_end_dt >= 5",
    ).index.tolist()
    print("eval size: ", len(eval_edinet_code_list_5))
    eval_response_5_df = response_tbl_all.query(
        "response_edinetCode in @eval_edinet_code_list_5 and period_end_dt >= @eval_periodend_startdate and period_end_dt <= @eval_periodend_enddate",
    )

    # ot計算上bs plどちらかの変化が0の場合は除くので、他もそれに合わせる(->calc_distance_clip.py)
    # ot_distance_list_df_kpi_2020 = load_ot_distance_list_kpi_2020()
    # docid_clean = list(set(ot_distance_list_df_kpi_2020.source_docid))

    response_train_pool = list(set(response_train.response_edinetCode))
    # eval term
    response_train_pool_year_num = (
        response_tbl_all.query(
            "response_edinetCode in @response_train_pool and period_end_dt >= @eval_periodend_startdate and period_end_dt <= @eval_periodend_enddate",
        )
        .groupby("response_edinetCode")
        .agg(
            {"period_end_dt": "nunique"},
        )
    )
    # train term last year

    mask = (
        response_tbl_all.response_edinetCode.isin(eval_edinet_code_list_5)
        & (
            response_tbl_all.period_end_dt + pd.DateOffset(years=1)
            > train_periodend_enddate
        )
        & (response_tbl_all.period_end_dt <= train_periodend_enddate)
    )
    eval_response_df = response_tbl_all.loc[mask]

    train_edinet_code_list_5 = response_train_pool_year_num.query(
        "period_end_dt >= 5",
    ).index.tolist()
    print("retrieval pool size: ", len(train_edinet_code_list_5))
    # eval term
    retrieval_pool_response_5_df = response_tbl_all.query(
        "response_edinetCode in @train_edinet_code_list_5 and period_end_dt >= @eval_periodend_startdate and period_end_dt <= @eval_periodend_enddate",
    )
    # train term last year
    mask = (
        response_tbl_all.response_edinetCode.isin(train_edinet_code_list_5)
        & (
            response_tbl_all.period_end_dt + pd.DateOffset(years=1)
            > train_periodend_enddate
        )
        & (response_tbl_all.period_end_dt <= train_periodend_enddate)
    )
    retrieval_pool_df = response_tbl_all.loc[mask]
    return (
        eval_response_df,  # 検索用（target） train termの最終年
        retrieval_pool_df,  # 検索用（retrieval pool） train termの最終年
        eval_response_5_df,  # targetの5年分のdocID
        retrieval_pool_response_5_df,  # retrieval poolの5年分のdocID
    )


# sector


class ResponseTblWithSector(pa.DataFrameModel):
    """Response tbl with year"""

    business_class_edinet: Series[str]
    business_class_tse: Series[str]
    business_class_tse_na: Series[int] = pa.Field(isin=[0, 1])
    year: Series[str]
    period_end_dt: Series[datetime]
    response_edinetCode: Series[str]
    response_secCode: Series[str]


class BusinessSector:
    def __init__(self):
        edinetcode = pd.read_csv(
            XBRL_PROJDIR / "data/0_metadata/dataset2003/EdinetcodeDlInfo1903.csv",
            header=1,
            index_col=False,
            engine="python",
            encoding="cp932",
            dtype={"証券コード": str},
        )
        edinetcode2 = pd.read_csv(
            XBRL_PROJDIR / "data/0_metadata/dataset2003/EdinetcodeDlInfo.csv",
            header=1,
            index_col=False,
            engine="python",
            encoding="cp932",
            dtype={"証券コード": str},
        )
        code_2020 = edinetcode.query("上場区分=='上場'")
        code_2020_2 = edinetcode2.query("上場区分=='上場'")

        edinetcode2023 = pd.read_csv(
            XBRL_PROJDIR / "data/0_metadata/trial/EdinetcodeDlInfo2312.csv",
            header=1,
            index_col=False,
            engine="python",
            encoding="cp932",
            dtype={"証券コード": str},
        )
        code_2023 = edinetcode2023.query("上場区分=='上場'")

        edinetcode2407 = pd.read_csv(
            XBRL_PROJDIR / "data/0_metadata/trial/EdinetcodeDlInfo2407.csv",
            header=1,
            index_col=False,
            engine="python",
            encoding="cp932",
            dtype={"証券コード": str},
        )
        code_2407 = edinetcode2407.query("上場区分=='上場'")

        edinetcode2507 = pd.read_csv(
            XBRL_PROJDIR / "data/0_metadata/dataset_2507/EdinetcodeDlInfo2507.csv",
            header=1,
            index_col=False,
            engine="python",
            encoding="cp932",
            dtype={"証券コード": str},
        )
        code_2507 = edinetcode2507.query("上場区分=='上場'")

        edinetinfo = (
            pd.concat([code_2020, code_2020_2, code_2023, code_2407, code_2507], axis=0)
            .reset_index(drop=True)
            .drop_duplicates(subset="ＥＤＩＮＥＴコード", keep="last")
        )
        self.edinetinfo = edinetinfo.assign(
            business_class_edinet=edinetinfo["提出者業種"],
        )

    def add_business_sector(
        self,
        df: pd.DataFrame,
        concat_colname: str = "response_edinetCode",
    ) -> ResponseTblWithSector:
        preproc_summary_smp_ans = pd.merge(
            df,
            self.edinetinfo[["ＥＤＩＮＥＴコード", "business_class"]],
            left_on=concat_colname,
            right_on="ＥＤＩＮＥＴコード",
            how="left",
        )
        return preproc_summary_smp_ans

    def add_business_sector_sec(
        self,
        df: pd.DataFrame,
        concat_colname: str = "response_edinetCode",
        concat_colname_sec: str = "response_secCode",
    ) -> pd.DataFrame:
        # concat edinet info for interpolate
        df_edinet_class = pd.merge(
            df,
            self.edinetinfo[["ＥＤＩＮＥＴコード", "business_class_edinet"]],
            left_on=concat_colname,
            right_on="ＥＤＩＮＥＴコード",
            how="left",
        )
        print(df_edinet_class.shape)
        # load tse sector info
        # https://www3.cuc.ac.jp/~tsuchiya/
        # filename_business_class="./Projects/XBRL_common_space_projection/data/0_metadata/external/tse_sector_2023-11-30.csv"
        # filename_business_class = "./Projects/XBRL_common_space_projection/data/0_metadata/external/tse_sector_2024-06-30.csv"
        filename_business_class = "./Projects/XBRL_common_space_projection/data/0_metadata/dataset_2507/tiba/tse_sector_2025-03-31.csv"

        business_class = pd.read_csv(
            filename_business_class,
            header=0,
            index_col=0,
            dtype={"sec_code": str},
        )
        # filter into latest
        business_class["date_ts"] = pd.to_datetime(business_class["date"])
        business_class_g = business_class.groupby(by="sec_code")
        business_class_latest = business_class.loc[
            business_class_g["date_ts"].idxmax(),
            :,
        ][["sec_code", "type_33"]]

        # rename business class name un consistence detected by followings
        #   a=set(business_class.type_33.value_counts().index)
        #   b=set(preproc_summary_smp_addclass.business_class.value_counts().index)
        #   print(a-b,b-a)
        business_class_latest["business_class_tse"] = (
            business_class_latest.type_33.replace(
                {"倉庫・運輸関連業": "倉庫・運輸関連"},
            )
        )
        # concat tse sector info
        df_tosyo_class = pd.merge(
            df_edinet_class,
            business_class_latest[["sec_code", "business_class_tse"]],
            left_on=concat_colname_sec,
            right_on="sec_code",
            how="left",
        )
        print(df_tosyo_class.shape)
        # replace '-' to nan
        df_tosyo_class["business_class_tse"] = df_tosyo_class[
            "business_class_tse"
        ].replace("-", np.nan)
        # label nan
        df_tosyo_class = df_tosyo_class.assign(
            business_class_tse_na=(df_tosyo_class.business_class_tse.isna() * 1).astype(
                int,
            ),
        )
        print(
            df_tosyo_class.business_class_tse_na.sum(),
            " documents are interpolated about those business class, by the EDINET-DL-Info data.",
        )
        # interpolate
        df_tosyo_class["business_class_tse"] = df_tosyo_class[
            "business_class_tse"
        ].fillna(df_tosyo_class["business_class_edinet"])
        print(df_tosyo_class["business_class_tse"].nunique())
        return df_tosyo_class[
            [
                "docID",
                "business_class_edinet",
                "business_class_tse",
                "business_class_tse_na",
                concat_colname,
                concat_colname_sec,
                "year",
                "period_end_dt",
            ]
        ]


# %%
def get_sector() -> ResponseTblWithSector:
    business_sector = BusinessSector()
    # test split
    filename_test = DATADIR / cfg["response_tbl_test"]
    response_test = pd.read_pickle(filename_test)
    response_test = response_test.query(
        "task_kpi_flg == 1",
    )  # .response_edinetCode.nunique()

    response_tbl_smp_addclass_test = business_sector.add_business_sector_sec(
        response_test.reset_index(),
    ).set_index("docID")  # .query("business_class_tse.notna()")
    print(len(response_tbl_smp_addclass_test.query("business_class_tse.notna()")))

    response_tbl_smp_addclass_test = response_tbl_smp_addclass_test.query(
        "business_class_tse.notna()",
    )
    # train split
    filename_train = DATADIR / cfg["response_tbl_train"]
    response_train = pd.read_pickle(filename_train)

    response_tbl_smp_addclass_train = business_sector.add_business_sector_sec(
        response_train.reset_index(),
    ).set_index("docID")  # .query("business_class_tse.notna()")
    print(len(response_tbl_smp_addclass_train.query("business_class_tse.notna()")))
    return response_tbl_smp_addclass_test, response_tbl_smp_addclass_train


# %%
def get_eval_docid_spec_list(test_docid: list[str]) -> list[str]:
    bs_data = load_bs_data(docid_list=test_docid)
    pl_data = load_pl_data(docid_list=test_docid)

    ind_spec_account_list = [
        "BNK",
        "CNS",
        "CNA",
        "SEC",
        "INS",
        "RWY",
        "WAT",
        "NWY",
        "telecommunications",
        "ELE",
        "GAS",
        "LIQ",
        "IVT",
        "INV",
        "SPF",
        "MED",
        "EDU",
        "CMD",
        "LEA",
        "FND",
    ]
    bs_data = bs_data.assign(
        ind_spec_account_flg=bs_data.key.str.contains("|".join(ind_spec_account_list)),
    )
    pl_data = pl_data.assign(
        ind_spec_account_flg=pl_data.key.str.contains("|".join(ind_spec_account_list)),
    )

    eval_docid_spec_list = list(
        set(bs_data.query("ind_spec_account_flg == 1").docid)
        | set(pl_data.query("ind_spec_account_flg == 1").docid),
    )
    return eval_docid_spec_list


def get_task_div_pred_sector(
    exclude_docid_spec_list: bool = False,
) -> list[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Get eval and retrieve(same year as eval) set
    Eval: 138
    Retrieval: 2000
    """
    response_tbl_all = get_all_response_tbl()

    filename = DATADIR / cfg["response_tbl_train"]
    response_train = pd.read_pickle(filename)

    filename = DATADIR / cfg["response_tbl_test"]
    response_test = pd.read_pickle(filename)

    business_sector = BusinessSector()
    response_tbl_smp_addclass = business_sector.add_business_sector_sec(
        response_tbl_all.reset_index(),
    ).set_index("docID")  # .query("business_class_tse.notna()")
    print(len(response_tbl_smp_addclass.query("business_class_tse.notna()")))

    response_tbl_with_sector = response_tbl_smp_addclass.query(
        "business_class_tse.notna()",
    )
    docid_with_sector = response_tbl_with_sector.index.tolist()

    response_test_pool = list(
        set(response_test.query("task_kpi_flg == 1").response_edinetCode),
    )

    # filter
    # 1. test split
    # 2. business_class exists
    # 3. year == 2024
    response_test_pool_year_df = response_tbl_all.query(
        "response_edinetCode in @response_test_pool and index in @docid_with_sector and period_end_dt >= '2024-04-01' and period_end_dt <= '2025-03-31'",
    )
    print(f"eval候補 (フィルタリング前): {len(response_test_pool_year_df)}")
    eval_sector_test = response_tbl_with_sector.query(
        "index in @response_test_pool_year_df.index",
    )

    # filter to retrieval pool
    # 1. train split
    # 2. business_class exists
    # 3. year == 2024
    response_train_pool = list(
        set(response_train.response_edinetCode),
    )
    response_train_pool_year_df = response_tbl_all.query(
        "response_edinetCode in @response_train_pool and index in @docid_with_sector and period_end_dt >= '2024-04-01' and period_end_dt <= '2025-03-31'",
    )
    print(f"train候補 (フィルタリング前): {len(response_train_pool_year_df)}")
    retrieval_pool_sector = response_tbl_with_sector.query(
        "index in @response_train_pool_year_df.index",
    )

    if exclude_docid_spec_list:
        # 業種特有の勘定科目を持つdocIDを除外（評価データ）
        eval_docid_spec_list = get_eval_docid_spec_list(eval_sector_test.index.tolist())
        print(f"eval_docid_spec_list (除外対象): {len(eval_docid_spec_list)}")
        eval_sector_test = eval_sector_test.query("index not in @eval_docid_spec_list")
        print(f"eval (業種特有除外後): {len(eval_sector_test)}")

        # 業種特有の勘定科目を持つdocIDを除外（訓練データ）
        train_docid_spec_list = get_eval_docid_spec_list(
            retrieval_pool_sector.index.tolist(),
        )
        print(f"train_docid_spec_list (除外対象): {len(train_docid_spec_list)}")
        retrieval_pool_sector = retrieval_pool_sector.query(
            "index not in @train_docid_spec_list",
        )
        print(f"train (業種特有除外後): {len(retrieval_pool_sector)}")

    # 共通のクラス（業種）のみを残す
    train_sector_set = set(retrieval_pool_sector.business_class_tse)
    eval_sector_set = set(eval_sector_test.business_class_tse)
    eval_sector_set = set([
        "情報・通信業",
        "サービス業",
        "小売業",
        "卸売業",
        "不動産業",
        "医薬品",
        "その他金融業",
        "化学",
        "電気機器",
        "その他製品",
        "機械",
        "食料品"
        ])
    common_sector = train_sector_set & eval_sector_set
    print(f"common_sector: {len(common_sector)}")
    assert len(common_sector) == 12, f"common_sector: {common_sector}"

    # 共通クラスでフィルタリング
    retrieval_pool_sector = retrieval_pool_sector.query(
        "business_class_tse in @common_sector",
    )
    print(f"train (共通クラスのみ): {len(retrieval_pool_sector)}")
    eval_sector_test = eval_sector_test.query(
        "business_class_tse in @common_sector",
    )
    print(f"eval (共通クラスのみ): {len(eval_sector_test)}")

    # 最終的なクラス分布を確認
    print(
        f"\n最終的な訓練データのクラス数: {retrieval_pool_sector.business_class_tse.nunique()}",
    )
    print(
        f"最終的な評価データのクラス数: {eval_sector_test.business_class_tse.nunique()}",
    )

    return (
        eval_sector_test,
        retrieval_pool_sector,
    )
