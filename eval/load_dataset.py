# %%
import pickle
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import pandera as pa
import yaml
from pandera.typing import Series
from pydantic.functional_validators import BeforeValidator

CFGDIR = Path("./Projects/t_interpretable_fs/src")
with (CFGDIR / "cfg_exp_main.yaml").open(encoding="utf-8") as _cfg_f:
    cfg = yaml.load(_cfg_f, Loader=yaml.FullLoader)

XBRL_PROJDIR = Path(cfg["xbrl_proj_path"])
DATADIR = Path(cfg["data_path"])
PROCDIR = Path(cfg["procdir_path"])


# %% Get response_tbl #########################################################
# Get response_tbl from XBRL_common_space_projection
#########################################################
def get_columns_df(schima: pa.DataFrameModel) -> list:
    return list(schima.to_schema().columns.keys())


StrOrNone = Annotated[str, BeforeValidator(lambda x: x or "")]


class ResponseTblWithYear(pa.DataFrameModel):
    """Response tbl with year"""

    response_edinetCode: Series[str]  # response edinet code
    response_periodStart: Series[str]  # response period start
    response_periodEnd: Series[str]  # response period end
    response_secCode: Series[str]  # response sec code
    year: Series[str]  # account taxonomy year
    response_filerName: Series[str]  # response filer name
    response_fundCode: Series[str]  # response fund code
    response_ordinanceCode: Series[str]  # response ordinance code
    response_formCode: Series[str]  # response form code
    response_docTypeCode: Series[str]  # response doc type code
    dataset: Series[str]  # dataset
    download_status: Series[str]  # download status
    download_data_path: Series[str]  # download data path
    period_end_dt: Series[datetime]  # response period end date


@pa.check_output(ResponseTblWithYear)
def get_all_response_tbl(include_wo_accdata: bool = False) -> ResponseTblWithYear:
    # - 2024 data
    filename = XBRL_PROJDIR / "data/3_processed/dataset_2407/response_tbl_with_year.pkl"
    response_tbl = pd.read_pickle(filename)

    # 2024 - 2025 data
    filename = (
        XBRL_PROJDIR / "data/3_processed/dataset_2507/response_tbl_with_year_add.pkl"
    )
    response_tbl_add = pd.read_pickle(filename)
    response_tbl_add = response_tbl_add.query("year != '-'")
    response_tbl = pd.concat([response_tbl, response_tbl_add])
    response_tbl["period_end_dt"] = pd.to_datetime(
        response_tbl["response_periodEnd"],
    )
    if not include_wo_accdata:
        df_amounts_change_all = pd.read_csv(PROCDIR / "df_amounts_change_all_0218.csv")
        df_amounts_change_all_docid = df_amounts_change_all.docid.tolist()
        response_tbl = response_tbl.query("index in @df_amounts_change_all_docid")

    return response_tbl


def get_fraud_res_tbl_all() -> pd.DataFrame:
    filename = (
        XBRL_PROJDIR
        / "data/3_processed/dataset_2507/restatement/response_tbl_fraud_all.pkl"
    )
    response_tbl_fraud = pd.read_pickle(filename)
    print(len(response_tbl_fraud))
    return response_tbl_fraud


def test_get_all_response_tbl():
    response_tbl_all = get_all_response_tbl()
    assert response_tbl_all.year.nunique() == 12
    assert response_tbl_all.response_edinetCode.nunique() > 5000


class BSData(pa.DataFrameModel):
    docid: Series[str]  # docid
    key: Series[str]  # key
    data_str: Series[str]  # data str
    data_num: Series[int]  # data num
    label_jp_long_filled: Series[str]  # label jp long filled
    diff: Series[float]  # diff
    diff_rate: Series[float]  # diff rate
    diff_rate_assets: Series[float]  # diff rate assets
    label_jp_long_filled_parent: Series[str] = pa.Field(nullable=True)
    sign_lab: Series[str] = pa.Field(nullable=True)  # sign lab
    calc_parent_key: Series[str] = pa.Field(nullable=True)  # calc parent key


class PLData(pa.DataFrameModel):
    docid: Series[str]  # docid
    key: Series[str]  # key
    data_str: Series[str]  # data str
    data_num: Series[int]  # data num
    label_jp_long_filled: Series[str] = pa.Field(nullable=True)  # label jp long filled
    diff: Series[float]  # diff
    diff_rate: Series[float] = pa.Field(nullable=True)  # diff rate
    diff_rate_assets: Series[float] = pa.Field(nullable=True)  # diff rate assets
    # label_jp_long_filled_parent: Series[str] = pa.Field(
    #    nullable=True
    # )  # label jp long filled parent
    sign_lab: Series[str] = pa.Field(nullable=True)  # sign lab
    calc_parent_key: Series[str] = pa.Field(nullable=True)  # calc parent key


@pa.check_output(BSData)
def load_bs_data(docid_list: list[str]) -> BSData:
    """v0817"""
    filename = DATADIR / "merged_bs_filled_unq.pkl"
    x_data = pd.read_pickle(filename)  # [["docid", "label_jp_long_filled_id", "diff"]]
    print(x_data.columns)
    x_data = x_data.query("docid in @docid_list").copy()
    x_data["data_num"] = x_data["data_str"].fillna(0).astype(int)
    x_data = x_data.merge(
        x_data[["docid", "key", "label_jp_long_filled"]].rename(
            columns={
                "label_jp_long_filled": "label_jp_long_filled_parent",
                "key": "key_p",
            },
        ),
        left_on=["calc_parent_key", "docid"],
        right_on=["key_p", "docid"],
        how="left",
    )[get_columns_df(BSData)]
    return x_data


@pa.check_output(PLData)
def load_pl_data(docid_list: list[str]) -> PLData:
    """v0817"""
    filename = DATADIR / "merged_pl_filled_unq.pkl"
    data_pl = pd.read_pickle(filename)
    data_pl = data_pl.query("docid in @docid_list").copy()
    data_pl["data_num"] = data_pl["data_str"].fillna(0).astype(int)
    data_pl = data_pl[get_columns_df(PLData)]
    return data_pl


@pa.check_output(BSData)
def load_bs_data_amd(docid_list: list[str]) -> BSData:
    """v0817"""
    filename = DATADIR / "restatement" / "merged_bs_filled_unq.pkl"
    x_data = pd.read_pickle(filename)  # [["docid", "label_jp_long_filled_id", "diff"]]
    print(x_data.columns)
    x_data = x_data.query("docid in @docid_list").copy()
    # 空文字列をNaNに変換してからfillnaで0に置換
    x_data["data_num"] = x_data["data_str"].replace("", None).fillna(0).astype(int)
    x_data["diff_rate"] = x_data["diff_rate"].fillna(0)
    x_data["diff_rate_assets"] = x_data["diff_rate_assets"].fillna(0)
    x_data = x_data.merge(
        x_data[["docid", "key", "label_jp_long_filled"]].rename(
            columns={
                "label_jp_long_filled": "label_jp_long_filled_parent",
                "key": "key_p",
            },
        ),
        left_on=["calc_parent_key", "docid"],
        right_on=["key_p", "docid"],
        how="left",
    )[get_columns_df(BSData)]
    return x_data


@pa.check_output(PLData)
def load_pl_data_amd(docid_list: list[str]) -> PLData:
    """v0817"""
    filename = DATADIR / "restatement" / "merged_pl_filled_unq.pkl"
    data_pl = pd.read_pickle(filename)
    data_pl = data_pl.query("docid in @docid_list").copy()
    # 空文字列をNaNに変換してからfillnaで0に置換
    data_pl["data_num"] = data_pl["data_str"].replace("", None).fillna(0).astype(int)
    data_pl["diff_rate"] = data_pl["diff_rate"].fillna(0)
    data_pl["diff_rate_assets"] = data_pl["diff_rate_assets"].fillna(0)
    data_pl = data_pl[get_columns_df(PLData)]
    return data_pl


def load_data_amd(docid_list: list[str]):
    filename = DATADIR / "restatement" / "merged_bs_filled_unq.pkl"
    x_data = pd.read_pickle(filename)  # [["docid", "label_jp_long_filled_id", "diff"]]
    print(x_data.columns)
    x_data = x_data.query("docid in @docid_list")
    x_data = x_data.merge(
        x_data[["docid", "key", "label_jp_long_filled"]].rename(
            columns={"label_jp_long_filled": "label_jp_long_filled_parent"},
        ),
        left_on=["calc_parent_key", "docid"],
        right_on=["key", "docid"],
        how="left",
    )[
        [
            "docid",
            "key_x",
            "data_str",
            "label_jp_long_filled",
            "diff",
            "diff_rate",
            "diff_rate_assets",
            "label_jp_long_filled_parent",
            "sign_lab",
            "calc_parent_key",
        ]
    ]
    filename = DATADIR / "restatement" / "merged_pl_filled.pkl"
    data_pl = pd.read_pickle(filename)

    data_pl["data_num"] = data_pl["data_str"].fillna(0).astype(int)
    return x_data, data_pl


# %%


class DataDbs:
    def __init__(self, x_data):
        with open(PROCDIR / "le" / "key_le.pkl", "rb") as f:
            self.le_account = pickle.load(f)

        self.x_data = x_data
        self.x_data = self.preprocess_dbs(x_data)

    def preprocess_dbs(self, x_data):
        x_data["exist_flg"] = x_data["key"].isin(self.le_account.classes_).astype(int)
        x_data_exist = x_data.query("exist_flg == 1")
        x_data_exist["key_id"] = self.le_account.transform(x_data_exist["key"])
        x_data_exist["key_id"] = x_data_exist["key_id"] + 1
        x_data = x_data.merge(
            x_data_exist[["key", "docid", "key_id"]],
            left_on=["docid", "key"],
            right_on=["docid", "key"],
            how="left",
        )
        return x_data

    def get_dataset(self, docid: str):
        d_bs = self.x_data.query("docid == @docid")
        d_bs = self.fill_key_id_dbs(d_bs)
        d_bs["diff_data_num_rate"] = d_bs["diff"].abs() / d_bs["diff"].abs().sum()

        return d_bs

    def get_dataset2(self, docid_list: list[str]):
        d_bs = self.x_data.query("docid in @docid_list")
        d_bs = self.fill_key_id_dbs(d_bs)
        d_bs["diff_data_num_rate"] = d_bs["diff"].abs() / d_bs["diff"].abs().sum()
        return d_bs

    def fill_key_id_dbs(self, d_bs):
        d_bs_fill_df = d_bs.query("key_id.notna()")[["key_id", "key"]]
        d_bs = d_bs.merge(
            d_bs_fill_df.rename(
                columns={"key_id": "key_id_fill", "key": "key_fill"},
            ),
            left_on="calc_parent_key",
            right_on="key_fill",
            how="left",
        )
        d_bs["key_id"] = d_bs["key_id"].fillna(d_bs["key_id_fill"])
        return d_bs


class DataPl:
    def __init__(self, data_pl):
        with open(PROCDIR / "le" / "key_le.pkl", "rb") as f:
            self.le_account = pickle.load(f)
        self.data_pl = self.preprocess_pl(data_pl)

    def preprocess_pl(self, data_pl):
        data_pl["exist_flg"] = data_pl["key"].isin(self.le_account.classes_).astype(int)
        data_pl_exist = data_pl.query("exist_flg == 1")
        data_pl_exist["key_id"] = self.le_account.transform(data_pl_exist["key"])
        data_pl_exist["key_id"] = data_pl_exist["key_id"] + 1

        data_pl = data_pl.merge(
            data_pl_exist[["key", "docid", "key_id"]],
            left_on=["docid", "key"],
            right_on=["docid", "key"],
            how="left",
        )
        return data_pl

    def get_dataset(self, docid: str):
        pl = self.data_pl.query("docid == @docid")
        pl = self.fill_key_id_pl(pl)
        pl["data_num_rate"] = pl["data_num"].abs() / pl["data_num"].abs().sum()
        return pl

    def get_dataset2(self, docid_list: list[str]):
        pl = self.data_pl.query("docid in @docid_list")
        pl = self.fill_key_id_pl(pl)
        pl["data_num_rate"] = pl["data_num"].abs() / pl["data_num"].abs().sum()
        return pl

    def fill_key_id_pl(self, pl):
        pl_fill_df = pl.query("key_id.notna()")[["key_id", "key"]]
        pl = pl.merge(
            pl_fill_df.rename(columns={"key_id": "key_id_fill", "key": "key_fill"}),
            left_on="calc_parent_key",
            right_on="key_fill",
            how="left",
        )
        pl["key_id"] = pl["key_id"].fillna(pl["key_id_fill"])
        pl["sign_lab"] = pl["sign_lab"].fillna("-")
        return pl


# %%
def test_load_bs_data():
    bs_data = load_bs_data(docid_list=["S1002006", "S10020AM"])
    pl_data = load_pl_data(docid_list=["S1002006", "S10020AM"])
    data_dbs = DataDbs(bs_data)
    data_pl = DataPl(pl_data)


# %%

# %%
# %%
