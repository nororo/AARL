"""
Fill missing account names in the financial statement data.

"""
import sys

import numpy as np
import pandas as pd

sys.path.append(r"./Projects/XBRL_common_space_projection")
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pickle
import sys

import pandera as pa
from pandera.typing import Series
from tqdm import tqdm

sys.path.append(r"./Projects/XBRL_common_space_projection")
sys.path.append(
    r"./Projects/XBRL_common_space_projection/src/edinet_xbrl_prep",
)
import os
from datetime import datetime

from edinet_xbrl_prep.link_base_file_analyzer import *
from src.data.libs.utils import DataLinageJson
from src.data.libs.xbrl_prep_patch import *

# %%


class FsDataDf(pa.DataFrameModel):
    # ParentChildLink
    """'key': taxonomy like 'jpcrp_cor:NetSales'
    'data_str': data (string) like '1000000'
    'decimals': 3桁の表示
    'precision': ???
    'context_ref': # T:-3, M:-6, B:-9
    'element_name':
    'unit': # JPY
    'period_type':
    'isTextBlock_flg':
    'abstract_flg':
    'period_start': # durationの場合 当期末日, instantの場合 None
    'period_end': # durationの場合 当期末日, instantの場合 当期末日
    'instant_date': # durationの場合 None, instantの場合 当期末日
    'end_date_pv': # durationの場合 前期末日, instantの場合 None
    'instant_date_pv': # durationの場合 None, instantの場合 前期対象日
    'scenario':# シナリオ
    'role': #
    'label_jp':
    'label_jp_long':
    'label_en':
    'label_en_long':
    'order':
    'child_key':
    'docid':

    calculated_flg=data.key.isin(calc_dict_all[role].keys()).astype(int),
            net_flg=data.key.isin(net_keys).astype(int),
            net_src_flg=data.key.isin(calc_src_keys).astype(int),
            calc_parent_key
    """

    key: Series[str] = pa.Field(nullable=True)
    data_str: Series[str] = pa.Field(nullable=True)
    decimals: Series[str] = pa.Field(nullable=True)
    # precision: Series[str] = pa.Field(nullable=True)
    context_ref: Series[str] = pa.Field(nullable=True)
    element_name: Series[str] = pa.Field(nullable=True)
    unit: Series[str] = pa.Field(nullable=True)
    period_type: Series[str] = pa.Field(
        isin=["instant", "duration"],
        nullable=True,
    )  # 'instant','duration'
    isTextBlock_flg: Series[int] = pa.Field(
        isin=[0, 1],
        nullable=True,
        coerce=True,
    )  # 0,1
    abstract_flg: Series[int] = pa.Field(isin=[0, 1], nullable=True, coerce=True)  # 0,1
    period_start: Series[str] = pa.Field(nullable=True)
    period_end: Series[str] = pa.Field(nullable=True)
    instant_date: Series[str] = pa.Field(nullable=True)
    # end_date_pv: Series[str] = pa.Field(nullable=True)
    # instant_date_pv: Series[str] = pa.Field(nullable=True)
    # scenario: Series[str] = pa.Field(nullable=True)
    role: Series[str] = pa.Field(nullable=True)
    label_jp: Series[str] = pa.Field(nullable=True)
    label_jp_long: Series[str] = pa.Field(nullable=True)
    label_en: Series[str] = pa.Field(nullable=True)
    label_en_long: Series[str] = pa.Field(nullable=True)
    order: Series[float] = pa.Field(nullable=True)
    order_str: Series[str] = pa.Field(nullable=True)
    layer: Series[str] = pa.Field(nullable=True)
    # child_key: Series[str] = pa.Field(nullable=True)
    parent_key_list: list
    docid: Series[str] = pa.Field(nullable=True)
    non_consolidated_flg: Series[int] = pa.Field(
        isin=[0, 1],
        nullable=True,
        coerce=True,
    )  # 0,1
    current_flg: Series[int] = pa.Field(isin=[0, 1], nullable=True, coerce=True)  # 0,1
    prior_flg: Series[int] = pa.Field(isin=[0, 1], nullable=True, coerce=True)  # 0,1
    calculated_flg: Series[int] = pa.Field(
        isin=[0, 1],
        nullable=True,
        coerce=True,
    )  # 0,1
    net_flg: Series[int] = pa.Field(isin=[0, 1], nullable=True, coerce=True)  # 0,1
    net_src_flg: Series[int] = pa.Field(isin=[0, 1], nullable=True, coerce=True)  # 0,1
    debit_flg: Series[int] = pa.Field(isin=[0, 1], nullable=True, coerce=True)  # 0,1
    credit_flg: Series[int] = pa.Field(isin=[0, 1], nullable=True, coerce=True)  # 0,1
    calc_parent_key: Series[str] = pa.Field(nullable=True)
    accounting_standards: Series[str] = pa.Field(nullable=True)


def PL_test(data_context: FsDataDf, calc_dict_role) -> dict:
    test_dict_acc_val_t = {}
    data_context.data_str = data_context.data_str.fillna("0")
    data_context.data_str = data_context.data_str.replace("", "0")
    pl_val = np.nan
    kpi_val = np.nan
    ordinary_income_val = np.nan
    pl_key = None
    kpi_key = None
    ordinary_income_key = None
    # pl_after_incomtax = [parent for parent,childs_dict in calc_dict_role.items() if 'jppfs_cor:IncomeTaxes' in childs_dict]
    # pl_key = pl_after_incomtax[0]
    pl_key1 = "jppfs_cor:ProfitLoss"
    pl_key2 = "jppfs_cor:NetIncome"
    if len(data_context.query("key == @pl_key1")) == 0:
        if len(data_context.query("key == @pl_key2")) == 0:
            test_dict_acc_val_t.update({"pl_exsist": False})
        else:
            pl_val = data_context.query("key == @pl_key2").data_str.astype(int).sum()
            pl_key = pl_key2
            test_dict_acc_val_t.update({"pl_exsist": True})
    else:
        pl_val = data_context.query("key == @pl_key1").data_str.astype(int).sum()
        pl_key = pl_key1
        test_dict_acc_val_t.update({"pl_exsist": True})

    if len(data_context.query("data_str.notna()")) > 0:
        kpi_key = (
            data_context.query("data_str.notna()")
            .sort_values("order")
            .head(1)
            .key.item()
        )
        if (len(data_context.query("key == @kpi_key")) == 0) | (kpi_key == None):
            test_dict_acc_val_t.update({"kpi_exsist": False})
        else:
            # display(data_context.query("key == @kpi_key"))
            kpi_val = data_context.query("key == @kpi_key").data_str.astype(int).sum()
            test_dict_acc_val_t.update({"kpi_exsist": True})
    else:
        test_dict_acc_val_t.update({"kpi_exsist": False})

    # print(json.dumps(calc_dict_role,indent=4))
    try:
        tmp_list = [
            (key, item) for key, item in calc_dict_role[pl_key].items() if item > 0
        ]
        pl_key2, _ = tmp_list[0]
        tmp_list = [
            (key, item)
            for key, item in calc_dict_role[pl_key2].items()
            if (item > 0) & ("Extra" not in key) & ("EI" not in key) & ("EL" not in key)
        ]
        if len(tmp_list) > 0:
            ordinary_income_key, _ = tmp_list[0]
        else:
            ordinary_income_key = "jppfs_cor:OrdinaryIncome"
    except Exception:
        ordinary_income_key = "jppfs_cor:OrdinaryIncome"
        # print(e)
    if (len(data_context.query("key == @ordinary_income_key")) == 0) | (
        ordinary_income_key == None
    ):
        test_dict_acc_val_t.update({"ordinary_income_exsist": False})
    else:
        ordinary_income_val = (
            data_context.query("key == @ordinary_income_key").data_str.astype(int).sum()
        )
        test_dict_acc_val_t.update({"ordinary_income_exsist": True})

    return pl_key, kpi_key, ordinary_income_key, pl_val, kpi_val, ordinary_income_val


PROJPATH = r"./Projects/XBRL_common_space_projection/"
PROJDIR = Path(PROJPATH)
TESTDIR = Path(PROJPATH) / "tests/20250127"
TESTDIR

# itr_docID = 'S100TY62'


# %%
# %%
def preproc_fs(
    itr_docID,
    response_tbl,
    calc_list_common_obj,
    account_label_common_obj_dict,
):
    temp_path_str = str(
        PROJDIR
        / "data"
        / "2_intermediate"
        / (f"data_pool_{response_tbl.loc[itr_docID, 'dataset']}")
        / itr_docID,
    )
    filename = temp_path_str + "/fs_tbl.pkl"
    fs_tbl = FsDataDf(pd.read_pickle(filename))
    log_filename = temp_path_str + "/fs_tbl_log.json"
    log_dict = json.load(open(log_filename))
    # print(json.dumps(log_dict, indent=4))

    # result
    preproc_log = {}

    # response_tbl
    preproc_log["docid"] = itr_docID
    preproc_log["edinet_code"] = response_tbl.loc[itr_docID, "response_edinetCode"]
    preproc_log["sec_code"] = response_tbl.loc[itr_docID, "response_secCode"]
    preproc_log["period_start"] = response_tbl.loc[itr_docID, "response_periodStart"]
    preproc_log["period_end"] = response_tbl.loc[itr_docID, "response_periodEnd"]
    preproc_log["EDINET_taxonomy_year"] = response_tbl.loc[itr_docID, "year"]

    # decide NonCons_flg
    if "record_cnt__rol_BalanceSheet" in log_dict["log_cnt_dict"]:
        if (
            "not_na_value__Prior1YearInstant_NonConsolidatedMember"
            in log_dict["log_cnt_dict"]["record_cnt__rol_BalanceSheet"]
        ):
            py_bs_NonConsolidated = log_dict["log_cnt_dict"][
                "record_cnt__rol_BalanceSheet"
            ]["not_na_value__Prior1YearInstant_NonConsolidatedMember"]
        else:
            py_bs_NonConsolidated = 0
        if (
            "not_na_value__CurrentYearInstant_NonConsolidatedMember"
            in log_dict["log_cnt_dict"]["record_cnt__rol_BalanceSheet"]
        ):
            cy_bs_NonConsolidated = log_dict["log_cnt_dict"][
                "record_cnt__rol_BalanceSheet"
            ]["not_na_value__CurrentYearInstant_NonConsolidatedMember"]
        else:
            cy_bs_NonConsolidated = 0
    else:
        py_bs_NonConsolidated = 0
        cy_bs_NonConsolidated = 0

    if "record_cnt__rol_ConsolidatedBalanceSheet" in log_dict["log_cnt_dict"]:
        if (
            "not_na_value__Prior1YearInstant"
            in log_dict["log_cnt_dict"]["record_cnt__rol_ConsolidatedBalanceSheet"]
        ):
            py_bs_Consolidated = log_dict["log_cnt_dict"][
                "record_cnt__rol_ConsolidatedBalanceSheet"
            ]["not_na_value__Prior1YearInstant"]
        else:
            py_bs_Consolidated = 0
        if (
            "not_na_value__CurrentYearInstant"
            in log_dict["log_cnt_dict"]["record_cnt__rol_ConsolidatedBalanceSheet"]
        ):
            cy_bs_Consolidated = log_dict["log_cnt_dict"][
                "record_cnt__rol_ConsolidatedBalanceSheet"
            ]["not_na_value__CurrentYearInstant"]
        else:
            cy_bs_Consolidated = 0
    else:
        py_bs_Consolidated = 0
        cy_bs_Consolidated = 0

    if "record_cnt__rol_StatementOfIncome" in log_dict["log_cnt_dict"]:
        # if log_dict["log_cnt_dict"].get("record_cnt__rol_StatementOfIncome") is not None:
        if (
            "not_na_value__Prior1YearDuration_NonConsolidatedMember"
            in log_dict["log_cnt_dict"]["record_cnt__rol_StatementOfIncome"]
        ):
            py_pl_NonConsolidated = log_dict["log_cnt_dict"][
                "record_cnt__rol_StatementOfIncome"
            ]["not_na_value__Prior1YearDuration_NonConsolidatedMember"]
        else:
            py_pl_NonConsolidated = 0
        if (
            "not_na_value__CurrentYearDuration_NonConsolidatedMember"
            in log_dict["log_cnt_dict"]["record_cnt__rol_StatementOfIncome"]
        ):
            cy_pl_NonConsolidated = log_dict["log_cnt_dict"][
                "record_cnt__rol_StatementOfIncome"
            ]["not_na_value__CurrentYearDuration_NonConsolidatedMember"]
        else:
            cy_pl_NonConsolidated = 0
    else:
        py_pl_NonConsolidated = 0
        cy_pl_NonConsolidated = 0

    if "record_cnt__rol_ConsolidatedStatementOfIncome" in log_dict["log_cnt_dict"]:
        if (
            "not_na_value__Prior1YearDuration"
            in log_dict["log_cnt_dict"]["record_cnt__rol_ConsolidatedStatementOfIncome"]
        ):
            py_pl_Consolidated = log_dict["log_cnt_dict"][
                "record_cnt__rol_ConsolidatedStatementOfIncome"
            ]["not_na_value__Prior1YearDuration"]
        else:
            py_pl_Consolidated = 0
        if (
            "not_na_value__CurrentYearDuration"
            in log_dict["log_cnt_dict"]["record_cnt__rol_ConsolidatedStatementOfIncome"]
        ):
            cy_pl_Consolidated = log_dict["log_cnt_dict"][
                "record_cnt__rol_ConsolidatedStatementOfIncome"
            ]["not_na_value__CurrentYearDuration"]
        else:
            cy_pl_Consolidated = 0
    else:
        py_pl_Consolidated = 0
        cy_pl_Consolidated = 0

    NonCons_num = (
        min(
            py_bs_NonConsolidated + py_pl_NonConsolidated,
            cy_bs_NonConsolidated + cy_pl_NonConsolidated,
        )
        * 0.2
    )
    Consolidated_th = min(
        py_bs_Consolidated + py_pl_Consolidated,
        cy_bs_Consolidated + cy_pl_Consolidated,
    )
    if NonCons_num > Consolidated_th:
        NonCons_flg = 1
        bs_role = "rol_BalanceSheet"
        pl_role = "rol_StatementOfIncome"
        py_bs_rolecontext_ref = "Prior1YearInstant_NonConsolidatedMember"
        py_pl_rolecontext_ref = "Prior1YearDuration_NonConsolidatedMember"
        cy_bs_rolecontext_ref = "CurrentYearInstant_NonConsolidatedMember"
        cy_pl_rolecontext_ref = "CurrentYearDuration_NonConsolidatedMember"
    else:
        NonCons_flg = 0
        bs_role = "rol_ConsolidatedBalanceSheet"
        pl_role = "rol_ConsolidatedStatementOfIncome"
        py_bs_rolecontext_ref = "Prior1YearInstant"
        py_pl_rolecontext_ref = "Prior1YearDuration"
        cy_bs_rolecontext_ref = "CurrentYearInstant"
        cy_pl_rolecontext_ref = "CurrentYearDuration"

    fs_tbl = fs_tbl.set_index("key")
    fs_tbl["taxonomy_year"] = preproc_log["EDINET_taxonomy_year"]

    for year in list(range(int(preproc_log["EDINET_taxonomy_year"]), 2014, -1)):
        year_str = str(year)
        account_label_common_obj_dict[year_str].get_assign_common_label()
        pre_account = account_label_common_obj_dict[
            year_str
        ].get_assign_common_label()  # .set_index("key")
        pre_account["taxonomy_year"] = year_str

        mask = (
            (fs_tbl.label_jp == "-")
            & (fs_tbl.label_jp_long == "-")
            & (fs_tbl.label_en == "-")
            & (fs_tbl.label_en_long == "-")
        )
        fs_tbl.loc[mask, "label_jp"] = None
        fs_tbl.loc[mask, "label_jp_long"] = None
        fs_tbl.loc[mask, "label_en"] = None
        fs_tbl.loc[mask, "label_en_long"] = None
        fs_tbl.loc[mask, "taxonomy_year"] = None
        missing_keys = fs_tbl.loc[mask].index

        fs_tbl.loc[mask, "label_jp"] = (
            fs_tbl.loc[mask, "label_jp"]
            .fillna(pre_account.query("index in @missing_keys")["label_jp"])
            .fillna("-")
        )
        fs_tbl.loc[mask, "label_jp_long"] = (
            fs_tbl.loc[mask, "label_jp_long"]
            .fillna(pre_account.query("index in @missing_keys")["label_jp_long"])
            .fillna("-")
        )
        fs_tbl.loc[mask, "label_en"] = (
            fs_tbl.loc[mask, "label_en"]
            .fillna(pre_account.query("index in @missing_keys")["label_en"])
            .fillna("-")
        )
        fs_tbl.loc[mask, "label_en_long"] = (
            fs_tbl.loc[mask, "label_en_long"]
            .fillna(pre_account.query("index in @missing_keys")["label_en_long"])
            .fillna("-")
        )
        fs_tbl.loc[mask, "taxonomy_year"] = fs_tbl.loc[mask, "taxonomy_year"].fillna(
            pre_account.query("index in @missing_keys")["taxonomy_year"],
        )
        fs_tbl["taxonomy_year"] = fs_tbl["taxonomy_year"].fillna("-")

    # fs_tbl["taxonomy_v"] = pd.DataFrame(fs_tbl.index.str.split(':',expand=True))[0]+ '_' + str(fs_tbl["taxonomy_year"])
    fs_tbl = fs_tbl.reset_index()

    keep_columns: list = [
        "data_str",
        "decimals",
        "context_ref",
        #'element_name', 'unit',
        #'period_type', 'isTextBlock_flg', 'abstract_flg', 'period_start',
        #'period_end', 'instant_date',
        "role",
        "label_jp",
        "label_jp_long",
        "label_en",
        "label_en_long",
        #'order',
        "order_str",
        "layer",
        "parent_key_list",
        "docid",
        #'non_consolidated_flg', 'current_flg', 'prior_flg',
        "calculated_flg",
        "net_flg",
        "net_src_flg",
        "debit_flg",
        "credit_flg",
        "calc_parent_key",
        "data",
        "taxonomy_year",
        #'accounting_standards'
    ]
    keep_columns_py = [
        "data",
    ]

    # extract bs and pl
    cy_pl = fs_tbl.query("role == @pl_role & context_ref == @cy_pl_rolecontext_ref")
    cy_bs = fs_tbl.query("role == @bs_role & context_ref == @cy_bs_rolecontext_ref")
    py_pl = fs_tbl.query("role == @pl_role & context_ref == @py_pl_rolecontext_ref")
    py_bs = fs_tbl.query("role == @bs_role & context_ref == @py_bs_rolecontext_ref")

    net_keys = [
        key
        for key in calc_list_common_obj.calc_dict_all[bs_role].items()
        if key[-3:] == "Net"
    ]
    calc_src_keys = [
        list(item.keys())
        for key, item in calc_list_common_obj.calc_dict_all[bs_role].items()
        if key[-3:] == "Net"
    ]
    flatten = lambda l: [item for sublist in l for item in sublist]
    calc_src_keys = flatten(calc_src_keys)

    # fill net_flg
    mask = cy_bs.net_src_flg == 0
    cy_bs.loc[mask, "net_src_flg"] = (
        cy_bs.loc[mask, "key"].isin(calc_src_keys).astype(int)
    )
    mask = cy_bs.net_flg == 0
    cy_bs.loc[mask, "net_flg"] = cy_bs.loc[mask, "key"].isin(net_keys).astype(int)
    mask = cy_bs.calculated_flg == 0
    cy_bs.loc[mask, "calculated_flg"] = (
        cy_bs.loc[mask, "key"]
        .isin(calc_list_common_obj.calc_dict_all[bs_role].keys())
        .astype(int)
    )

    mask = py_bs.net_src_flg == 0
    py_bs.loc[mask, "net_src_flg"] = (
        py_bs.loc[mask, "key"].isin(calc_src_keys).astype(int)
    )
    mask = py_bs.net_flg == 0
    py_bs.loc[mask, "net_flg"] = py_bs.loc[mask, "key"].isin(net_keys).astype(int)
    mask = py_bs.calculated_flg == 0
    py_bs.loc[mask, "calculated_flg"] = (
        py_bs.loc[mask, "key"]
        .isin(calc_list_common_obj.calc_dict_all[bs_role].keys())
        .astype(int)
    )

    cy_pl.calc_parent_key = cy_pl.calc_parent_key.fillna(
        cy_pl.apply(calc_list_common_obj.parent_by_common_calc, role=pl_role, axis=1),
    )
    cy_bs.calc_parent_key = cy_bs.calc_parent_key.fillna(
        cy_bs.apply(calc_list_common_obj.parent_by_common_calc, role=bs_role, axis=1),
    )
    py_pl.calc_parent_key = py_pl.calc_parent_key.fillna(
        py_pl.apply(calc_list_common_obj.parent_by_common_calc, role=pl_role, axis=1),
    )
    py_bs.calc_parent_key = py_bs.calc_parent_key.fillna(
        py_bs.apply(calc_list_common_obj.parent_by_common_calc, role=bs_role, axis=1),
    )

    # fill label

    # cy_bs
    # fill by common calc
    mask = (cy_bs.debit_flg + cy_bs.credit_flg) != 1

    cy_bs.loc[mask, "debit_flg"] = (
        cy_bs.loc[mask, "key"]
        .apply(lambda x: x in calc_list_common_obj.bs_debit_list)
        .astype(int)
    )
    cy_bs.loc[mask, "credit_flg"] = (
        cy_bs.loc[mask, "key"]
        .apply(lambda x: x in calc_list_common_obj.bs_credit_list)
        .astype(int)
    )

    # fill by parent
    mask = (cy_bs.debit_flg + cy_bs.credit_flg) != 1
    # cy_bs.loc[mask,"debit_flg"] = cy_bs.loc[mask,:].apply(lambda sr: cy_bs.query("key == @sr.calc_parent_key").debit_flg.sum(),axis=1)
    # cy_bs.loc[mask,"credit_flg"] = cy_bs.loc[mask,:].apply(lambda sr: cy_bs.query("key == @sr.calc_parent_key").credit_flg.sum(),axis=1)

    # py_bs
    # fill by common calc
    mask = (py_bs.debit_flg + py_bs.credit_flg) != 1
    py_bs.loc[mask, "debit_flg"] = (
        py_bs.loc[mask, "key"]
        .apply(lambda x: x in calc_list_common_obj.bs_debit_list)
        .astype(int)
    )
    py_bs.loc[mask, "credit_flg"] = (
        py_bs.loc[mask, "key"]
        .apply(lambda x: x in calc_list_common_obj.bs_credit_list)
        .astype(int)
    )

    # cy_pl
    # fill by common calc
    mask = (cy_pl.debit_flg + cy_pl.credit_flg) != 1
    cy_pl.loc[mask, "debit_flg"] = (
        cy_pl.loc[mask, "key"]
        .apply(lambda x: x in calc_list_common_obj.pl_debit_list)
        .astype(int)
    )
    cy_pl.loc[mask, "credit_flg"] = (
        cy_pl.loc[mask, "key"]
        .apply(lambda x: x in calc_list_common_obj.pl_credit_list)
        .astype(int)
    )

    # py_pl
    # fill by common calc
    mask = (py_pl.debit_flg + py_pl.credit_flg) != 1
    py_pl.loc[mask, "debit_flg"] = (
        py_pl.loc[mask, "key"]
        .apply(lambda x: x in calc_list_common_obj.pl_debit_list)
        .astype(int)
    )
    py_pl.loc[mask, "credit_flg"] = (
        py_pl.loc[mask, "key"]
        .apply(lambda x: x in calc_list_common_obj.pl_credit_list)
        .astype(int)
    )

    if log_dict["calc_dict_all"].get(pl_role) is None:
        calc_dict_all_pl = calc_list_common_obj.calc_dict_all[pl_role]
    else:
        calc_dict_all_pl = log_dict["calc_dict_all"][pl_role]
    pl_key, kpi_key, ordinary_income_key, cy_pl_val, cy_kpi_val, cy_ord_val = PL_test(
        cy_pl,
        calc_dict_all_pl,
    )
    py_pl_key, py_kpi_key, py_ordinary_income_key, py_pl_val, py_kpi_val, py_ord_val = (
        PL_test(py_pl, calc_dict_all_pl)
    )

    # calc_dict_all_pl = calc_list_common_obj.calc_dict_all[pl_role]
    # calc_dict_all_pl.update(log_dict["calc_dict_all"][pl_role])

    kpi_val = {
        "pl_key": pl_key,
        "kpi_key": kpi_key,
        "ordinary_income_key": ordinary_income_key,
        "py_pl_key": py_pl_key,
        "py_kpi_key": py_kpi_key,
        "py_ordinary_income_key": py_ordinary_income_key,
        "cy_pl_val": cy_pl_val,
        "cy_kpi_val": cy_kpi_val,
        "cy_ord_val": cy_ord_val,
        "py_pl_val": py_pl_val,
        "py_kpi_val": py_kpi_val,
        "py_ord_val": py_ord_val,
        # "cy_asset_val":cy_bs_kpi_val['assets_val'],
        # "cy_net_assets_val":cy_bs_kpi_val['net_assets_val'],
        # "cy_liabilities_val":cy_bs_kpi_val['liabilities_val'],
        # "cy_lib_and_netassets_val":cy_bs_kpi_val['lib_and_netassets_val'],
        # "py_asset_val":py_bs_kpi_val['assets_val'],
        # "py_net_assets_val":py_bs_kpi_val['net_assets_val'],
        # "py_liabilities_val":py_bs_kpi_val['liabilities_val'],
        # "py_lib_and_netassets_val":py_bs_kpi_val['lib_and_netassets_val'],
    }
    # kpi
    # py_kpi_val_sr = pd.Series(log_dict["log_kpi_dict"][pl_role]["kpi__"+py_pl_rolecontext_ref])
    # cy_kpi_val_sr = pd.Series(log_dict["log_kpi_dict"][pl_role]["kpi__"+cy_pl_rolecontext_ref])

    if log_dict["log_kpi_dict"].get(bs_role) is None:
        py_pl_kpi_val = None
    elif log_dict["log_kpi_dict"][bs_role].get("kpi__" + py_bs_rolecontext_ref) is None:
        py_bs_kpi_val = None
    else:
        py_bs_kpi_val = log_dict["log_kpi_dict"][bs_role][
            "kpi__" + py_bs_rolecontext_ref
        ]
        kpi_val["py_asset_val"] = py_bs_kpi_val["assets_val"]
        kpi_val["py_net_assets_val"] = py_bs_kpi_val["net_assets_val"]
        kpi_val["py_liabilities_val"] = py_bs_kpi_val["liabilities_val"]
        kpi_val["py_lib_and_netassets_val"] = py_bs_kpi_val["lib_and_netassets_val"]

    # calc diff
    if (
        log_dict["log_kpi_dict"].get(bs_role) is None
        or log_dict["log_kpi_dict"][bs_role].get("kpi__" + cy_bs_rolecontext_ref)
        is None
    ):
        cy_bs_kpi_val = None
    else:
        cy_bs_kpi_val = log_dict["log_kpi_dict"][bs_role][
            "kpi__" + cy_bs_rolecontext_ref
        ]
        kpi_val["cy_asset_val"] = cy_bs_kpi_val["assets_val"]
        kpi_val["cy_net_assets_val"] = cy_bs_kpi_val["net_assets_val"]
        kpi_val["cy_liabilities_val"] = cy_bs_kpi_val["liabilities_val"]
        kpi_val["cy_lib_and_netassets_val"] = cy_bs_kpi_val["lib_and_netassets_val"]

    if (len(py_bs) == 0) | (len(cy_bs) == 0):
        merged_bs = pd.DataFrame()
    else:
        merged_bs = pd.merge(
            preproc_num(cy_bs).set_index("key")[keep_columns],
            preproc_num(py_bs).set_index("key")[["data"]],
            on="key",
            suffixes=("_cy", "_py"),
            how="outer",
        )
        merged_bs["diff"] = merged_bs.data_cy.fillna(0) - merged_bs.data_py.fillna(0)
        merged_bs["deb_cred_uk_flg"] = (
            (merged_bs.debit_flg + merged_bs.credit_flg) != 1
        ).astype(int)
        if cy_bs_kpi_val != None:
            merged_bs["offset"] = cy_bs_kpi_val["assets_val"]
            merged_bs["diff_rate_assets"] = (
                merged_bs["diff"] / cy_bs_kpi_val["assets_val"]
            )
            merged_bs["diff_rate"] = merged_bs["diff"] / (
                merged_bs.data_py.fillna(0) + 1
            )

    if (len(py_pl) == 0) | (len(cy_pl) == 0):
        merged_pl = pd.DataFrame()
    else:
        merged_pl = pd.merge(
            preproc_num(cy_pl).set_index("key")[keep_columns],
            preproc_num(py_pl).set_index("key")[["data"]],
            on="key",
            suffixes=("_cy", "_py"),
            how="outer",
        )
        merged_pl["diff"] = merged_pl.data_cy.fillna(0) - merged_pl.data_py.fillna(0)
        merged_pl["deb_cred_uk_flg"] = (
            (merged_pl.debit_flg + merged_pl.credit_flg) != 1
        ).astype(int)
        if cy_bs_kpi_val != None:
            merged_pl["offset"] = cy_bs_kpi_val["assets_val"]
            merged_pl["diff_rate_assets"] = (
                merged_pl["diff"] / cy_bs_kpi_val["assets_val"]
            )
            merged_pl["diff_rate"] = merged_pl["diff"] / (
                merged_pl.data_py.fillna(0) + 1
            )

    # preproc_log["chk_kpi_pl_key"] = pl_key
    # preproc_log["chk_kpi_kpi_key"] = kpi_key
    # preproc_log["chk_kpi_ordinary_income_key"] = ordinary_income_key
    preproc_log["chk_kpi_pl_key_consistency"] = pl_key == py_pl_key
    preproc_log["chk_kpi_kpi_key_consistency"] = kpi_key == py_kpi_key
    preproc_log["chk_kpi_ordinary_income_key_consistency"] = (
        ordinary_income_key == py_ordinary_income_key
    )

    # term
    term_end = cy_pl.period_end.str.extract(r"(\d{4})-(\d{2})-(\d{2})")
    term_start = cy_pl.period_start.str.extract(r"(\d{4})-(\d{2})-(\d{2})")
    preproc_log["chk_term_len"] = (
        pd.to_datetime(term_end.sum(axis=1)) - pd.to_datetime(term_start.sum(axis=1))
    ).dt.days.min()
    # other info
    if len(cy_bs) == 0:
        preproc_log["decimals"] = None
        preproc_log["chk_unit"] = None
        preproc_log["chk_accounting_standards"] = None
    else:
        preproc_log["decimals"] = cy_bs.decimals.value_counts().index[0]
        preproc_log["chk_unit"] = cy_bs.unit.value_counts().index[0]
        preproc_log["chk_accounting_standards"] = (
            cy_bs.accounting_standards.value_counts().index[0]
        )

    if log_dict["test_dict_acc_val"].get(bs_role) is None:
        preproc_log["chk_py_pl_cons"] = False
    elif log_dict["test_dict_acc_val"][bs_role].get(
        "test_bs__" + py_bs_rolecontext_ref,
    ) is None or (
        log_dict["test_dict_acc_val"][bs_role]["test_bs__" + py_bs_rolecontext_ref].get(
            "cons_assets_liabilities",
        )
        is None
    ) | (
        log_dict["test_dict_acc_val"][bs_role]["test_bs__" + py_bs_rolecontext_ref].get(
            "cns_liabilities_net_assets",
        )
        is None
    ):
        preproc_log["chk_py_bs_cons"] = False
    else:
        preproc_log["chk_py_bs_cons"] = (
            (
                log_dict["test_dict_acc_val"][bs_role][
                    "test_bs__" + py_bs_rolecontext_ref
                ]["cons_assets_liabilities"]
            )
            & (
                log_dict["test_dict_acc_val"][bs_role][
                    "test_bs__" + py_bs_rolecontext_ref
                ]["cns_liabilities_net_assets"]
            )
        )
    if (
        log_dict["test_dict_acc_val"].get(bs_role) is None
        or log_dict["test_dict_acc_val"][bs_role].get(
            "test_bs__" + cy_bs_rolecontext_ref,
        )
        is None
        or (
            log_dict["test_dict_acc_val"][bs_role][
                "test_bs__" + cy_bs_rolecontext_ref
            ].get("cons_assets_liabilities")
            is None
        )
        | (
            log_dict["test_dict_acc_val"][bs_role][
                "test_bs__" + cy_bs_rolecontext_ref
            ].get("cns_liabilities_net_assets")
            is None
        )
    ):
        preproc_log["chk_cy_bs_cons"] = False
    else:
        preproc_log["chk_cy_bs_cons"] = (
            (
                log_dict["test_dict_acc_val"][bs_role][
                    "test_bs__" + cy_bs_rolecontext_ref
                ]["cons_assets_liabilities"]
            )
            & (
                log_dict["test_dict_acc_val"][bs_role][
                    "test_bs__" + cy_bs_rolecontext_ref
                ]["cns_liabilities_net_assets"]
            )
        )

    preproc_log["chk_cy_pl_cal_rate"] = (
        (cy_pl.calc_parent_key != "not_common_account")
        & (cy_pl.calc_parent_key != "top_account")
    ).sum() / len(cy_pl)
    preproc_log["chk_py_pl_cal_rate"] = (
        (py_pl.calc_parent_key != "not_common_account")
        & (py_pl.calc_parent_key != "top_account")
    ).sum() / len(py_pl)
    preproc_log["chk_cy_bs_cal_rate"] = (
        (cy_bs.calc_parent_key != "not_common_account")
        & (cy_bs.calc_parent_key != "top_account")
    ).sum() / len(cy_bs)
    preproc_log["chk_py_bs_cal_rate"] = (
        (py_bs.calc_parent_key != "not_common_account")
        & (py_bs.calc_parent_key != "top_account")
    ).sum() / len(py_bs)

    return (
        preproc_log,
        kpi_val,
        merged_bs.reset_index(),
        merged_pl.reset_index(),
        cy_pl,
        cy_bs,
    )


# %% prep
def prep_account_list_common_obj(out_dir: Path):
    account_list_common_obj_dict = {}
    for year_str in [
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
    ]:
        account_list_common_obj_dict[year_str] = account_list_common_cor(
            data_path=out_dir,
            account_list_year=year_str,
        )
    out_filename = out_dir / "account_list_common_obj_dict.pkl"
    with open(out_filename, "wb") as f:
        pickle.dump(account_list_common_obj_dict, f)


def prep_calc_list(out_dir: Path):
    calc_list_common_obj_dict = {}
    for year in [
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
    ]:
        calc_list_common_obj = calc_list_common(out_dir, year)
        calc_list_common_obj.parse_calc_all()
        calc_list_common_obj_dict[year] = calc_list_common_obj

    out_filename = out_dir / "calc_list_common_obj.pkl"
    with open(out_filename, "wb") as f:
        pickle.dump(calc_list_common_obj_dict, f)


# %% main
def main():
    account_list_dir = PROJDIR / "data/0_metadata/common" / "account_list"
    # prep_account_list_common_obj(out_dir=account_list_dir)
    filename = account_list_dir / "account_list_common_obj_dict.pkl"
    with open(filename, "rb") as f:
        account_list_common_obj_dict = pickle.load(f)

    prep_calc_list(out_dir=account_list_dir)
    filename = account_list_dir / "calc_list_common_obj.pkl"
    with open(filename, "rb") as f:
        calc_list_common_obj_dict = pickle.load(f)

    filename = PROJPATH + "data/3_processed/dataset_2407/response_tbl_with_year.pkl"
    response_tbl = pd.read_pickle(filename)

    filename = PROJPATH + "data/3_processed/dataset_2507/response_tbl_with_year_add.pkl"
    response_tbl_add = pd.read_pickle(filename)
    response_tbl_add = response_tbl_add.query("year != '-'")
    assert len(set(response_tbl_add.index) & set(response_tbl.index)) == 0

    response_tbl = pd.concat([response_tbl, response_tbl_add])

    log_dict_list = []
    kpi_val_list = []
    merged_bs_list = []
    merged_pl_list = []
    cy_pl_list = []
    cy_bs_list = []
    for itr_docID in tqdm(response_tbl.index):
        year = response_tbl.loc[itr_docID, "year"]
        preproc_log, kpi_val, merged_bs, merged_pl, cy_pl, cy_bs = preproc_fs(
            itr_docID,
            response_tbl,
            calc_list_common_obj_dict[year],
            account_list_common_obj_dict,
        )
        log_dict_list.append(preproc_log)
        kpi_val_list.append(kpi_val)
        merged_bs_list.append(merged_bs)
        merged_pl_list.append(merged_pl)
        cy_pl_list.append(cy_pl)
        cy_bs_list.append(cy_bs)

    # kpiがdebitの場合を削除
    # debitフラグを他の会社で補間
    # .query("net_src_flg ==0 and deb_cred_uk_flg ==0")

    pd.concat(merged_bs_list).reset_index(drop=True).to_pickle(
        PROJDIR / "data/3_processed/dataset_2507" / "merged_bs.pkl",
    )
    pd.concat(merged_pl_list).reset_index(drop=True).to_pickle(
        PROJDIR / "data/3_processed/dataset_2507" / "merged_pl.pkl",
    )
    pd.concat(cy_pl_list).reset_index(drop=True).to_pickle(
        PROJDIR / "data/3_processed/dataset_2507" / "cy_pl.pkl",
    )
    pd.concat(cy_bs_list).reset_index(drop=True).to_pickle(
        PROJDIR / "data/3_processed/dataset_2507" / "cy_bs.pkl",
    )
    pd.DataFrame(log_dict_list).to_pickle(
        PROJDIR / "data/3_processed/dataset_2507" / "preproc_log.pkl",
    )
    pd.DataFrame(kpi_val_list).to_pickle(
        PROJDIR / "data/3_processed/dataset_2507" / "kpi_val.pkl",
    )
    # make lin
    df = pd.concat(merged_bs_list).reset_index(drop=True)
    make_lin(
        file_path=PROJDIR / "data/3_processed/dataset_2507" / "merged_bs.pkl",
        df=df,
    )


def make_lin(file_path, df):
    assertion_text = """
    """
    processing_text = """
    calculation commonによる補間
    別年度account listによるlabelの補間
    BS差分, PL差分の算出, KPI抽出
    """
    header_note_txt = """
    """
    # file_path = out_filename
    ts_str = datetime.fromtimestamp(os.path.getctime(file_path)).strftime(
        "%Y-%m-%d %H:%M:%S",
    )
    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    DataLinageJson_obj = DataLinageJson(
        create_date=f"{ts_str}",
        check_date=f"{ts_now}",
        size=f"{os.path.getsize(file_path):,}",
        file_path=str(file_path),
        reader="""
            read_pickle
        """,
        encoding="utf-8",
        input_data={
            "fs_tbl.pkl": [
                str(
                    PROJDIR
                    / "data"
                    / "2_intermediate"
                    / "data_pool_{response_tbl.loc[itr_docID,'dataset']}"
                    / "itr_docID",
                ),
            ],
        },
        input_data_providing_func={},
        index_name=df.index.name,
        header=list(df.columns),
        count=len(df),
        unique_count_index=df.index.nunique(),
        unique_count_header=df.describe(include="all").T["unique"].to_dict(),
        example_rcd=df.iloc[0].to_dict(),
        header_note=header_note_txt,
        src="data/07_prep_fs_data.py",
        assertion="",
        processing=processing_text,
        note="",
    )
    DataLinageJson_obj.save()


# %%


# %%


def preproc_num(org_data: pd.DataFrame) -> pd.DataFrame:
    mask = org_data.parent_key_list.apply(
        lambda x: ("jppfs_cor:LiabilitiesAndNetAssets" in x)
        | ("jppfs_cor:BalanceSheetLineItems" in x)
        | ("jppfs_cor:NetAssetsAbstract" in x)
        | ("jppfs_cor:AssetsAbstract" in x),
    )
    org_data = org_data[~mask]
    org_data["data_str"] = org_data["data_str"].fillna("0").replace("", "0")
    org_data["data"] = pd.to_numeric(org_data["data_str"], errors="coerce")
    org_data["data"] = org_data.data.astype(
        "float",
    )  # for case of data is string object (cannot read csv with dtype=int)
    # org_data.data = org_data.data.fillna(-1)
    # org_data.data = org_data.data.replace(0,np.nan).fillna(0.5*10**(org_data['decimals'].astype(float)*-1))
    # org_data.data = org_data.data.replace(-1,0)
    return org_data


# %% dev calc


class calc_list_common:
    def __init__(self, data_path: str, account_list_year: str):
        linkfiles_dict = {
            "pre.xml": "jpcrp030000-asr",
            "lab.xml": "jpcrp",
            "lab-en.xml": "jpcrp",
        }
        schima_word_list = ["jppfs", "jpcrp"]
        self.taxonomy_file = data_path / f"taxonomy_{account_list_year}.zip"
        self.account_list_year = account_list_year
        self.temp_path = data_path / "tmp/taxonomy"
        self.temp_path.mkdir(parents=True, exist_ok=True)
        self.taxonomy_path = data_path / ("taxonomy_" + str(account_list_year))
        self.taxonomy_path.mkdir(parents=True, exist_ok=True)

        # self._download_taxonomy()
        self.path_jppfs_cal_list = self._download_jppfs_cal()
        self._build()

    def _download_taxonomy(self):
        download_link_dict = {
            "2025": "https://www.fsa.go.jp/search/20241112/1c_Taxonomy.zip",
            "2024": "https://www.fsa.go.jp/search/20231211/1c_Taxonomy.zip",
            "2023": "https://www.fsa.go.jp/search/20221108/1c_Taxonomy.zip",
            "2022": "https://www.fsa.go.jp/search/20211109/1c_Taxonomy.zip",
            "2021": "https://www.fsa.go.jp/search/20201110/1c_Taxonomy.zip",
            "2020": "https://www.fsa.go.jp/search/20191101/1c_Taxonomy.zip",
            "2019": "https://www.fsa.go.jp/search/20190228/1c_Taxonomy.zip",
            "2018": "https://www.fsa.go.jp/search/20180228/1c_Taxonomy.zip",
            "2017": "https://www.fsa.go.jp/search/20170228/1c.zip",
            "2016": "https://www.fsa.go.jp/search/20160314/1c.zip",
            "2015": "https://www.fsa.go.jp/search/20150310/1c.zip",
            "2014": "https://www.fsa.go.jp/search/20140310/1c.zip",
        }

        r = requests.get(download_link_dict[self.account_list_year], stream=True)
        with self.taxonomy_file.open(mode="wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

    def _download_jppfs_cal(self) -> list:
        already_download_list = list(self.taxonomy_path.glob("jppfs*_cal_*.xml"))

        if len(already_download_list) > 50:  # 652 files in 2024
            # print("already_download_list: ",len(already_download_list))
            return already_download_list
        with ZipFile(str(self.taxonomy_file)) as zf:
            fn = [
                item
                for item in zf.namelist()
                if ("_cal_" in item) & ("jppfs" in item) & ("dep" not in item)
            ]
            if len(fn) > 0:
                for f in fn:
                    zf.extract(f, self.temp_path)
        f_path_new_list = []
        for f_path in list(self.temp_path.glob("**/*.xml")):
            f_path_new = f_path.rename(self.taxonomy_path / f_path.name)
            f_path_new_list.append(f_path_new)
        # print("{} files are downloaded".format(len(f_path_new_list)))
        return f_path_new_list

    def _build(self):
        # self.get_label_common_obj_jpcrp_lab = get_calc_edge_list_common(
        #    file_str=self.path_jppfs_cal,
        #    temp_path_str=str(self.temp_path)
        #    )
        self.calc_list = []
        for path in self.path_jppfs_cal_list:
            get_calc_edge_list_common_obj = get_calc_edge_list_common(
                file_str=path,
            )
            calc_edge_df = get_calc_edge_list_common_obj.export_parent_child_link_df()
            calc_edge_df["name"] = path.name
            self.calc_list.append(calc_edge_df)

    def parse_calc_all(self):
        calc_dict_all = {}
        calc_parent_dict_all = {}
        bs_debit_list = []
        bs_credit_list = []
        pl_debit_list = []
        pl_credit_list = []
        for calc_edge_df in self.calc_list:
            (
                calc_dict_all_t,
                calc_parent_dict_all_t,
                bs_debit_list_t,
                bs_credit_list_t,
                pl_debit_list_t,
                pl_credit_list_t,
            ) = parse_calc(calc_edge_df)
            # calc_dict_all = calc_dict_all + calc_dict_all_t
            for role_common in calc_parent_dict_all_t:
                role = role_common.replace("_std_", "_")
                if calc_dict_all.get(role) is None:
                    calc_dict_all[role] = calc_dict_all_t[role_common]
                else:
                    for key in calc_dict_all_t[role_common]:
                        if calc_dict_all[role].get(key) is None:
                            calc_dict_all[role][key] = calc_dict_all_t[role_common][key]
                        else:
                            calc_dict_all[role][key].update(
                                calc_dict_all_t[role_common][key],
                            )

                if calc_parent_dict_all.get(role) is None:
                    calc_parent_dict_all[role] = calc_parent_dict_all_t[role_common]
                else:
                    calc_parent_dict_all[role].update(
                        calc_parent_dict_all_t[role_common],
                    )
            bs_debit_list = bs_debit_list + bs_debit_list_t
            bs_credit_list = bs_credit_list + bs_credit_list_t
            pl_debit_list = pl_debit_list + pl_debit_list_t
            pl_credit_list = pl_credit_list + pl_credit_list_t
        self.calc_dict_all = calc_dict_all
        self.calc_parent_dict_all = calc_parent_dict_all
        self.bs_debit_list = list(set(bs_debit_list))
        self.bs_credit_list = list(set(bs_credit_list))
        self.pl_debit_list = list(set(pl_debit_list))
        self.pl_credit_list = list(set(pl_credit_list))
        # return calc_dict_all,calc_parent_dict_all,bs_debit_list,bs_credit_list,pl_debit_list,pl_credit_list

    def parent_by_common_calc(self, sr: pd.Series, role: str):
        calc_parent_key = sr.calc_parent_key
        child_key = sr.key
        top_account_list = [
            "jppfs_cor:LiabilitiesAndNetAssets",
            "jppfs_cor:Assets",
            "jppfs_cor:ProfitLoss",
        ]
        if child_key in top_account_list:
            return "top_account"
        if self.calc_parent_dict_all[role].get(child_key) is None:
            return "not_common_account"
        return self.calc_parent_dict_all[role][child_key]


class get_calc_edge_list_common:
    def __init__(self, file_str: str):
        self.log_dict = {
            "is_cal_file_flg": 0,
            "get_cal_status": None,
            "get_cal_error_message": None,
        }
        self.file_path = Path(file_str)

        self.parse_cal_file()

    def extruct_cal_file_from_xbrlzip(self, zip_file_str):
        try:
            with ZipFile(str(zip_file_str)) as zf:
                fn = [
                    item
                    for item in zf.namelist()
                    if ("_cal_" in item) & ("jppfs" in item) & ("dep" not in item)
                ]
                if len(fn) > 0:
                    for f in fn:
                        zf.extract(f, self.temp_path)
                    # zf.extract(fn[0], self.temp_path)
            f_path_new_list = []
            for f_path in list(self.temp_path.glob("**/*.xml")):
                f_path_new = f_path.rename(self.taxonomy_path / f_path.name)
                f_path_new_list.append(f_path_new)

        except Exception as e:
            print(e)
            self.log_dict["is_cal_file_flg"] = 0
            self.log_dict["get_cal_status"] = "failure"
            self.log_dict["get_cal_error_message"] = str(e)

    def parse_cal_file(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        locators = []
        arcs = []
        for child in root:
            attr_sr_p = pd.Series(child.attrib)
            role = attr_sr_p[attr_sr_p.index.str.contains("role")].item()
            for child_of_child in child:
                locator = {"schima_taxonomi": None, "label": None, "role": role}
                arc = {
                    "parent": None,
                    "child": None,
                    "child_order": None,
                    "weight": None,
                    "role": role,
                }
                attr_sr = pd.Series(child_of_child.attrib)
                attr_type = attr_sr[attr_sr.index.str.contains("type")].item()
                if attr_type == "locator":
                    locator["schima_taxonomi"] = (
                        attr_sr[attr_sr.index.str.contains("href")].item().split("#")[1]
                    )
                    locator["label"] = attr_sr[
                        attr_sr.index.str.contains("label")
                    ].item()
                    locators.append(Locator(**locator))

                elif attr_type == "arc":
                    arc["parent"] = attr_sr[attr_sr.index.str.contains("from")].item()
                    arc["child"] = attr_sr[attr_sr.index.str.contains("to")].item()
                    arc["child_order"] = attr_sr[
                        attr_sr.index.str.contains("order")
                    ].item()
                    arc["weight"] = attr_sr[attr_sr.index.str.contains("weight")].item()
                    arcs.append(CalArc(**arc))

        self.locators = locators
        self.arcs = arcs

    def _make_label_to_taxonomi_dict(self):
        locators_df = pd.DataFrame(
            [locator.model_dump() for locator in self.locators],
        ).dropna(subset=["schima_taxonomi"])
        locators_df = locators_df.assign(
            role=locators_df.role.str.split("/", expand=True).iloc[:, -1],
            key=locators_df.schima_taxonomi.apply(format_taxonomi),
        )
        self.label_to_taxonomi_dict = locators_df.set_index("label")["key"].to_dict()

    def export_account_list_df(self) -> OriginalAccountList:
        locators_df = pd.DataFrame(
            [locator.model_dump() for locator in self.locators],
        ).dropna(subset=["schima_taxonomi"])
        locators_df = locators_df.assign(
            role=locators_df.role.str.split("/", expand=True).iloc[:, -1],
            key=locators_df.schima_taxonomi.apply(format_taxonomi),
        )
        cal_detail_list = OriginalAccountList(
            locators_df[get_columns_df(OriginalAccountList)],
        )
        return cal_detail_list

    def export_parent_child_link_df(self) -> CalParentChildLink:
        self._make_label_to_taxonomi_dict()
        arcs_df = pd.DataFrame([arc.model_dump() for arc in self.arcs]).dropna(
            subset=["child"],
        )
        arcs_df = arcs_df.assign(
            parent_key=arcs_df.parent.replace(self.label_to_taxonomi_dict),
            child_key=arcs_df.child.replace(self.label_to_taxonomi_dict),
            weight=arcs_df.weight.astype(float),
        )
        arcs_df = CalParentChildLink(
            arcs_df.drop_duplicates(subset=["parent_key", "child_key"])[
                get_columns_df(CalParentChildLink)
            ],
        )
        return arcs_df

    def export_log(self) -> GetCalLog:
        return GetCalLog(**self.log_dict)


# %%
def calculate_profit_loss_effect(
    account: str,
    relationships: dict,
    memo: dict = None,
    target_account: str = "jppfs_cor:ProfitLoss",
) -> int:
    """特定の勘定科目がProfitLossに与える影響（1: 増加, -1: 減少）を計算します

    Args:
        account (str): 勘定科目名
        relationships (dict): 勘定科目間の関係を表す辞書
        memo (dict, optional): メモ化用の辞書

    Returns:
        int: ProfitLossへの影響 (1: 増加, -1: 減少, 0: 影響なし)

    """
    if memo is None:
        memo = {}

    # メモ化による循環参照の防止
    if account in memo:
        return memo[account]

    # 初期値を設定
    memo[account] = 0

    # ProfitLossへの直接的な影響を確認
    if target_account in relationships:
        if account in relationships[target_account]:
            return relationships[target_account][account]

    # 親アカウントを探して影響を計算
    for parent, children in relationships.items():
        if account in children:
            parent_effect = calculate_profit_loss_effect(
                parent,
                relationships,
                memo,
                target_account,
            )
            memo[account] = parent_effect * children[account]
            return memo[account]

    return 0


def create_profit_loss_impact_dict(
    relationships: dict,
    target_account: str = "jppfs_cor:ProfitLoss",
) -> dict:
    """すべての勘定科目についてProfitLossへの影響を計算し、辞書として返します
    Args:
        relationships (dict): 勘定科目間の関係を表す辞書
    Returns:
        dict: 各勘定科目のProfitLossへの影響を表す辞書
    """
    # すべての勘定科目を収集
    all_accounts = set()
    for parent, children in relationships.items():
        all_accounts.add(parent)
        all_accounts.update(children.keys())

    # 各勘定科目の影響を計算
    result = {}
    for account in all_accounts:
        result[account] = calculate_profit_loss_effect(
            account,
            relationships,
            target_account=target_account,
        )

    return result


def parse_calc(calc_edge_df: pd.DataFrame):
    calc_dict_all = {}
    calc_parent_dict_all = {}
    bs_debit_list = ["jppfs_cor:Assets"]
    bs_credit_list = ["jppfs_cor:LiabilitiesAndNetAssets"]
    pl_debit_list = []
    pl_credit_list = ["jppfs_cor:ProfitLoss"]
    role_list = calc_edge_df.role.value_counts().index
    for role in role_list:
        role_suffix = role.split("/")[-1]
        p_key_set = set(calc_edge_df.query("role == @role").parent_key)
        len(p_key_set)
        calc_dict = {}
        calc_parent_dict = {}
        for p_key in p_key_set:
            calc_df = (
                calc_edge_df.query("role == @role and parent_key in @p_key")
                .sort_values("parent_key")
                .set_index("child_key")
            )
            calc_dict_t = {p_key: calc_df.weight.astype(int).to_dict()}
            calc_parent_dict_t = (
                calc_df.parent_key.to_dict()
            )  # key: child_key, value: parent_key
            calc_dict.update(calc_dict_t)
            calc_parent_dict.update(calc_parent_dict_t)

        calc_dict_all.update({role_suffix: calc_dict})
        # calc_parent_dict_all.update({role_suffix:calc_parent_dict})
        if calc_parent_dict_all.get(role_suffix) is None:
            calc_parent_dict_all[role_suffix] = calc_parent_dict
        else:
            calc_parent_dict_all[role_suffix].update(calc_parent_dict)
        # get profit loss impact
        relationships = calc_dict
        if ("_BalanceSheet" in role) or ("_ConsolidatedBalanceSheet" in role):
            result_dict_debit = create_profit_loss_impact_dict(
                relationships,
                target_account="jppfs_cor:Assets",
            )
            bs_debit_list = bs_debit_list + [
                key for key in result_dict_debit.keys() if result_dict_debit[key] > 0
            ]
            result_dict_credit = create_profit_loss_impact_dict(
                relationships,
                target_account="jppfs_cor:LiabilitiesAndNetAssets",
            )
            bs_credit_list = bs_credit_list + [
                key for key in result_dict_credit.keys() if result_dict_credit[key] > 0
            ]
        if ("_StatementOfIncome" in role) or ("_ConsolidatedStatementOfIncome" in role):
            result_dict_profit = create_profit_loss_impact_dict(
                relationships,
                target_account="jppfs_cor:ProfitLoss",
            )
            pl_debit_list = pl_debit_list + [
                key for key in result_dict_profit.keys() if result_dict_profit[key] < 0
            ]
            pl_credit_list = pl_credit_list + [
                key for key in result_dict_profit.keys() if result_dict_profit[key] > 0
            ]
    return (
        calc_dict_all,
        calc_parent_dict_all,
        bs_debit_list,
        bs_credit_list,
        pl_debit_list,
        pl_credit_list,
    )


if __name__ == "__main__":
    print("start")
    main()
    print("end")

# %%
# filename = PROJPATH + "data/3_processed/dataset_2507/response_tbl_with_year_add.pkl"
# response_tbl_add = pd.read_pickle(filename)
# response_tbl_add = response_tbl_add.query("year != '-'")
## %%
# response_tbl_add.download_status.value_counts()
# %%
# %%
