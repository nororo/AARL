# %%
import sys

import numpy as np
import pandas as pd

sys.path.append(r"/Users/noro/Documents/Projects/XBRL_common_space_projection")
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import sys

import pandera as pa
from pandera.typing import Series
from tqdm import tqdm

sys.path.append(r"/Users/noro/Documents/Projects/XBRL_common_space_projection")
sys.path.append(
    r"/Users/noro/Documents/Projects/XBRL_common_space_projection/src/edinet_xbrl_prep",
)

from edinet_xbrl_prep.link_base_file_analyzer import *
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


PROJPATH = r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR = Path(PROJPATH)
TESTDIR = Path(PROJPATH) / "tests/20250127"
TESTDIR

# itr_docID = 'S100TY62'


def preproc_fs(
    itr_docID,
    response_tbl,
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
    roles = list(set(fs_tbl.role))
    uns = fs_tbl.query(
        "key.str.contains('MajorComponentsOfSellingGeneralAndAdministrativeExpensesTextBlock')",
    )

    return roles, uns


# %%
filename = PROJPATH + "data/3_processed/dataset_2407/response_tbl_with_year.pkl"
response_tbl = pd.read_pickle(filename)

filename = PROJPATH + "data/3_processed/dataset_2507/response_tbl_with_year_add.pkl"
response_tbl_add = pd.read_pickle(filename)
response_tbl_add = response_tbl_add.query("year != '-'")
assert len(set(response_tbl_add.index) & set(response_tbl.index)) == 0

response_tbl = pd.concat([response_tbl, response_tbl_add])

uns_list = []
for itr_docID in tqdm(response_tbl.sample(500).index):
    _, uns = preproc_fs(itr_docID, response_tbl)
    if len(uns) > 0:
        print(itr_docID)
        uns_list.append(uns)
# %%
print(uns_list[0]["data_str"].values[0])
# %%
uns_df = pd.concat(uns_list)

# %%
uns_df.to_pickle(PROJPATH + "data/3_processed/dataset_2507/uns_df.pkl")
# %%
uns_df = pd.read_pickle(PROJPATH + "data/3_processed/dataset_2507/uns_df.pkl")
# %%
uns_df.head()
# %%
