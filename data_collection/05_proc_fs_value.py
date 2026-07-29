"""Processing text
env: proj_xbrl_dev

main()
    xbrl_proc()
        get_presentation_account_list()
            p_edges_schima
            original_account_list_schima

        account_link_tracer()
            get_child_keys()
            get_child_items()
            get_parent_keys()
            get_parent_items()
            get_role()
            get_child_keys_recursive()
            get_child_items_recursive()
            get_parent_keys_trace()
            search_keys()
            get_child_order_recursive_list()

        get_xbrl_rapper()
            - check if already parsed
            - get xbrl data from zip file
            get_xbrl_df()
                - parse xbrf by arrele
                get_fact_data() # extract fact data from arrele object
        get_xbrl_dei_df()

        get_data()
            preproc_nlp_xbrltext()
                data_utils.RtnDroper()
            multi_stage_chunk()
                CharacterTextSplitter()


"""

# %%
import sys

import numpy as np
import pandas as pd

sys.path.append(r"./Projects/XBRL_common_space_projection")


import json
import pickle
import shutil
import warnings
from pathlib import Path

import joblib
import pandera as pa
from pandera.typing import Series
from tqdm import tqdm

warnings.filterwarnings("ignore")


# %%
import sys

sys.path.append(r"./Projects/XBRL_common_space_projection")
sys.path.append(
    r"./Projects/XBRL_common_space_projection/src/edinet_xbrl_prep",
)

from edinet_xbrl_prep.fs_tbl import FsDataDf
from edinet_xbrl_prep.link_base_file_analyzer import *
from src.data.libs.xbrl_prep_patch import *

# import src.data.libs.xbrl_prep_patch as xbrl_prep_patch
# import importlib
# importlib.reload()


PROJPATH = r"./Projects/XBRL_common_space_projection/"
PROJDIR = Path(PROJPATH)
TESTDIR = Path(PROJPATH) / "tests/20250115"

# %%
# filename = TESTDIR / "account_list_common_obj_dict.pkl"
# with open(filename, 'rb') as f:
#    account_list_common_obj_dict = pickle.load(f)
#
## %%
# for year in account_list_common_obj_dict.keys():
#    with open(PROJDIR/ "data/0_metadata/common/taxonomy" / f"account_list_common_obj_dict_{year}.pkl", 'wb') as f:
#        pickle.dump(account_list_common_obj_dict[year], f)


# %% dataset
def load_response_tbl():
    filename = (
        PROJDIR
        / "data/0_metadata/dataset_2507/response_tbl_rst_2507_add_v250803_chk.csv"
    )
    chk_df = pd.read_csv(filename)
    # chk_df_lin = read_data_linage(filename)

    failure_docid_set = set(chk_df.docid)
    filename = PROJDIR / "data/3_processed/dataset_2507/response_tbl_with_year_add.pkl"
    response_tbl = pd.read_pickle(filename)  # .head()
    response_tbl = response_tbl.query("year != '-'")
    # response_tbl_lin = read_data_linage(filename)
    print(len(response_tbl))  # 43994
    response_tbl = response_tbl.query("index in @failure_docid_set")
    print(len(response_tbl))  # 43977
    return response_tbl


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


import time


@contextlib.contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.2f} s ")


import contextlib


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


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


# %%
def PL_test(data_context: FsDataDf, calc_dict_role) -> dict:
    test_dict_acc_val_t = {}
    data_context.data_str = data_context.data_str.fillna("0").replace("", "0")
    pl_val = np.nan
    kpi_val = np.nan
    ordinary_income_val = np.nan
    # pl_after_incomtax = [parent for parent,childs_dict in calc_dict_role.items() if 'jppfs_cor:IncomeTaxes' in childs_dict]
    # pl_key = pl_after_incomtax[0]
    pl_key1 = "jppfs_cor:ProfitLoss"
    pl_key2 = "jppfs_cor:NetIncome"
    if len(data_context.query("key == @pl_key1")) == 0:
        if len(data_context.query("key == @pl_key2")) == 0:
            test_dict_acc_val_t.update({"pl_exsist": False})
        else:
            pl_val = data_context.query("key == @pl_key2").data_str.astype(int).sum()
            test_dict_acc_val_t.update({"pl_exsist": True})
    else:
        pl_val = data_context.query("key == @pl_key1").data_str.astype(int).sum()
        test_dict_acc_val_t.update({"pl_exsist": True})

    kpi_key = (
        data_context.query("data_str.notna()")
        .sort_values("order_str")
        .head(1)
        .key.item()
    )
    if (len(data_context.query("key == @kpi_key")) == 0) | (kpi_key == None):
        test_dict_acc_val_t.update({"kpi_exsist": False})
    else:
        kpi_val = data_context.query("key == @kpi_key").data_str.astype(int).sum()
        test_dict_acc_val_t.update({"kpi_exsist": True})

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
            ordinary_income_key = None
    except Exception:
        ordinary_income_key = None
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

    return test_dict_acc_val_t, pl_val, kpi_val, ordinary_income_val


def BS_test(data_context: FsDataDf) -> dict:
    assets_val = np.nan
    liabilities_val = np.nan
    net_assets_val = np.nan
    lib_and_netassets_val = np.nan
    test_dict_acc_val_t = {}
    data_context.data_str = data_context.data_str.fillna("0").replace("", "0")
    assets_key = "jppfs_cor:Assets"
    if len(data_context.query("key == @assets_key")) == 0:
        # print("assets are zero in context: ",c_ref)
        test_dict_acc_val_t.update({"assets_exsist": False})
    else:
        assets_val = data_context.query("key == @assets_key").data_str.astype(int).sum()
        test_dict_acc_val_t.update({"assets_exsist": True})

    lib_and_netassets = "jppfs_cor:LiabilitiesAndNetAssets"
    if len(data_context.query("key == @lib_and_netassets")) == 0:
        # print("lib_and_netassets are zero in context: ",c_ref)
        test_dict_acc_val_t.update({"lib_and_netassets_exsist": False})
    else:
        lib_and_netassets_val = (
            data_context.query("key == @lib_and_netassets").data_str.astype(int).sum()
        )
        lib_and_netassets_val_decimals = (
            data_context.query("key == @lib_and_netassets").decimals.astype(int).iloc[0]
        )
        test_dict_acc_val_t.update({"lib_and_netassets_exsist": True})

    liabilities_key = "jppfs_cor:Liabilities"
    if len(data_context.query("key == @liabilities_key")) == 0:
        # print("liabilities are zero in context: ",c_ref)
        test_dict_acc_val_t.update({"liabilities_exsist": False})
    else:
        liabilities_val = (
            data_context.query("key == @liabilities_key").data_str.astype(int).sum()
        )
        test_dict_acc_val_t.update({"liabilities_exsist": True})

    net_assets_key = "jppfs_cor:NetAssets"
    if len(data_context.query("key == @net_assets_key")) == 0:
        # print("net assets are zero in context: ",c_ref)
        test_dict_acc_val_t.update({"net_assets_exsist": False})
    else:
        net_assets_val = (
            data_context.query("key == @net_assets_key").data_str.astype(int).sum()
        )
        test_dict_acc_val_t.update({"net_assets_exsist": True})

    if (
        test_dict_acc_val_t["assets_exsist"]
        & test_dict_acc_val_t["lib_and_netassets_exsist"]
    ):
        if assets_val != lib_and_netassets_val:
            test_dict_acc_val_t.update({"cons_assets_liabilities": False})
        else:
            test_dict_acc_val_t.update({"cons_assets_liabilities": True})

    if (
        test_dict_acc_val_t["liabilities_exsist"]
        & test_dict_acc_val_t["net_assets_exsist"]
        & test_dict_acc_val_t["lib_and_netassets_exsist"]
    ):
        if (
            abs(liabilities_val + net_assets_val - lib_and_netassets_val)
            > 10 ** (-1 * lib_and_netassets_val_decimals) * 2
        ):
            print(
                "lib_and_netassets: ",
                abs(liabilities_val + net_assets_val - lib_and_netassets_val),
            )
            test_dict_acc_val_t.update({"cns_liabilities_net_assets": False})
        else:
            test_dict_acc_val_t.update({"cns_liabilities_net_assets": True})
    return (
        test_dict_acc_val_t,
        assets_val,
        liabilities_val,
        net_assets_val,
        lib_and_netassets_val,
    )


def check_linkbasefile_obj(linkbasefile_obj):
    test_dict = {}
    for role in set(linkbasefile_obj.parent_child_df.role):
        p_key_set = set(
            linkbasefile_obj.parent_child_df.query("role == @role").parent_key,
        )
        # print(len(p_key_set))
        c_key_set = set(
            linkbasefile_obj.parent_child_df.query("role == @role").child_key,
        )
        # print(len(c_key_set))
        role_suffix = role.split("/")[-1]
        all_key_set = set(
            linkbasefile_obj.account_list.query("role == @role_suffix").key,
        )
        # print(len(all_key_set))
        test_rst = 0
        if len(p_key_set - all_key_set) != 0:
            test_rst = 1
            print(
                f"parent key in arc-link that is not included in locator: \n{p_key_set - all_key_set!s}",
            )
        if len(c_key_set - all_key_set) != 0:
            test_rst = 2
            print(
                f"child key in arc-link that is not included in locator: \n{p_key_set - all_key_set!s}",
            )
        # print(len(set(self.account_list.label)))
        if (
            len(
                set(linkbasefile_obj.label_tbl_jp.query("role == @role").key)
                - all_key_set,
            )
            != 0
        ):
            test_rst = 3
            print(
                f"key in label that is not included in locator: \n{set(self.label_tbl_jp.key) - all_key_set!s}",
            )
        test_dict.update({role_suffix: test_rst})
    return test_dict


# %%
def get_fs_tbl_new(
    account_list_common_obj,
    docid: str,
    zip_file_str: str,
    temp_path_str: str,
    role_keyward_list: list,
    update_flg: bool = False,
) -> FsDataDf:
    out_filename = temp_path_str + "/fs_tbl.pkl"
    log_out_filename = temp_path_str + "/fs_tbl_log.json"
    if (
        Path(out_filename).exists()
        & Path(log_out_filename).exists()
        & (update_flg == False)
    ):
        return 0
        # return FsDataDf(pd.read_pickle(out_filename)),json.load(open(log_out_filename))
    warnings.filterwarnings("ignore")
    # arrelle setting
    arelle_temp_dir = Path(PROJDIR / "Arelle_tmp")
    arelle_dir = arelle_temp_dir.joinpath(f"arelle_{docid}")
    arelle_dir.mkdir(exist_ok=True)
    cmd_arelle_dir = "--xdgConfigHome=" + str(arelle_dir)
    sys.argv.append(cmd_arelle_dir)

    # log through the process
    log_dict = {}
    log_cnt_dict = {}

    # get linkbase file
    linkbasefile_obj = linkbasefile_cor(
        zip_file_str=zip_file_str,
        temp_path_str=temp_path_str,
    )
    linkbasefile_obj.read_linkbase_file()

    # log num
    log_cnt_dict["account_list key"] = len(linkbasefile_obj.account_list.key)
    log_cnt_dict["unique_account_list key"] = len(
        set(linkbasefile_obj.account_list.key),
    )
    log_cnt_dict["role_count"] = (
        linkbasefile_obj.account_list.role.value_counts().to_dict()
    )

    # check
    log_dict["test_link_dict"] = check_linkbasefile_obj(linkbasefile_obj)
    mask = linkbasefile_obj.account_list[["key", "role"]].duplicated(keep=False)
    # print("differences in the label were ignored occurred the role: ",linkbasefile_obj.account_list[mask].sort_values("key").role.value_counts().to_dict())
    log_dict["label_ignored_dict"] = (
        linkbasefile_obj.account_list[mask]
        .sort_values("key")
        .role.value_counts()
        .to_dict()
    )

    # get calc_dict
    calc_dict_all = {}
    calc_parent_dict_all = {}
    bs_debit_list = ["jppfs_cor:Assets"]
    bs_credit_list = ["jppfs_cor:LiabilitiesAndNetAssets"]
    pl_debit_list = []
    pl_credit_list = ["jppfs_cor:ProfitLoss"]
    for role in set(linkbasefile_obj.parent_child_df.role):
        role_suffix = role.split("/")[-1]
        p_key_set = set(linkbasefile_obj.calc_edge_df.query("role == @role").parent_key)
        len(p_key_set)
        calc_dict = {}
        calc_parent_dict = {}
        for p_key in p_key_set:
            calc_df = (
                linkbasefile_obj.calc_edge_df.query(
                    "role == @role and parent_key in @p_key",
                )
                .sort_values("parent_key")
                .set_index("child_key")
            )
            calc_dict_t = {p_key: calc_df.weight.astype(int).to_dict()}
            calc_parent_dict_t = calc_df.parent_key.to_dict()
            calc_dict.update(calc_dict_t)
            calc_parent_dict.update(calc_parent_dict_t)

        calc_dict_all.update({role_suffix: calc_dict})
        calc_parent_dict_all.update({role_suffix: calc_parent_dict})

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

    log_dict["calc_dict_all"] = calc_dict_all
    log_cnt_dict["original_account_num"] = len(set(linkbasefile_obj.label_tbl_jp.key))

    # make account label
    linkbasefile_obj.make_account_label(
        account_list_common_obj=account_list_common_obj,
        role_list=role_keyward_list,
    )

    # parse xbrl
    xbrl_data_df, accounting_standards_dei, log_dict_xbrl = get_xbrl_rapper_loc(
        docid=docid,
        zip_file=zip_file_str,
        temp_dir=Path(temp_path_str),
        out_path=Path(temp_path_str),
        update_flg=False,
        xbrl_parsed_fname="textblock_cur.csv",  # "xbrl_parsed.csv"
    )
    shutil.rmtree(arelle_dir, ignore_errors=True)
    sys.argv.remove(cmd_arelle_dir)
    xbrl_data_df["accounting_standards"] = accounting_standards_dei
    log_dict["xbrl_data_df"] = log_dict_xbrl
    log_cnt_dict["xbrl_keys"] = len(set(xbrl_data_df.key))

    # merge and add columns
    data_list = []
    log_concat = {}
    test_dict_acc_val = {}
    log_kpi_dict = {}
    for role in list(linkbasefile_obj.account_tbl_role_dict.keys()):
        role_suffix = role.split("/")[-1]
        key_in_the_role: pd.Series = linkbasefile_obj.account_tbl_role_dict[role].key
        data = pd.merge(
            xbrl_data_df.query("key in @key_in_the_role"),
            linkbasefile_obj.account_tbl_role_dict[role],
            on="key",
            how="left",
        )

        net_keys = [key for key in calc_dict_all[role].items() if key[-3:] == "Net"]
        calc_src_keys = [
            list(item.keys())
            for key, item in calc_dict_all[role].items()
            if key[-3:] == "Net"
        ]
        flatten = lambda l: [item for sublist in l for item in sublist]
        calc_src_keys = flatten(calc_src_keys)

        data = data.assign(
            docid=docid,
            role=role,
            calculated_flg=data.key.isin(calc_dict_all[role].keys()).astype(int),
            net_flg=data.key.isin(net_keys).astype(int),
            net_src_flg=data.key.isin(calc_src_keys).astype(int),
            calc_parent_key=data.key.map(calc_parent_dict_all[role]),
        )

        data = data.assign(
            non_consolidated_flg=data.context_ref.str.contains(
                "NonConsolidated",
            ).astype(int),
            current_flg=data.context_ref.str.contains("CurrentYear").astype(int),
            prior_flg=data.context_ref.str.contains("Prior1Year").astype(int),
            debit_flg=(data.key.isin(pl_debit_list + bs_debit_list)).astype(int),
            credit_flg=(data.key.isin(pl_credit_list + bs_credit_list)).astype(int),
        )

        # fill na
        data["label_jp"] = data.label_jp.fillna("-")
        data["label_jp_long"] = data.label_jp_long.fillna("-")
        data["label_en"] = data.label_en.fillna("-")
        data["label_en_long"] = data.label_en_long.fillna("-")
        # print("concat_rate of",role,":",(data.label_jp!='-').sum()/len(data))
        log_concat[role] = (data.label_jp != "-").sum() / len(data)
        data = data.query(
            "(not (non_consolidated_flg==1 and role.str.contains('_Consolidated'))) and (not (non_consolidated_flg==0 and (not role.str.contains('_Consolidated') and not (role.str.contains('_CabinetOfficeOrdinanceOnDisclosure')))))",
        )

        # get summary value for (role x context ref)
        c_ref_cnt = data.context_ref.value_counts()
        c_ref_cnt = c_ref_cnt[c_ref_cnt >= 3]  # assets, liabilities, net assets
        c_ref_list = c_ref_cnt.index.to_list()
        # c_ref_list = [c_ref for c_ref in c_ref_list if '_' not in c_ref]
        test_dict_acc_val_role = {}
        log_cnt_dict_role = {}
        log_kpi_dict_role = {}
        for c_ref in c_ref_list:
            data_context = data.query("context_ref == @c_ref")
            log_cnt_dict_role.update(
                {"not_na_value__" + c_ref: int(data_context.data_str.notna().sum())},
            )
            log_cnt_dict_role.update({"len_data__" + c_ref: len(data_context)})

            # BS test
            if ("_BalanceSheet" in role) or ("_ConsolidatedBalanceSheet" in role):
                (
                    test_dict_acc_val_t,
                    assets_val,
                    liabilities_val,
                    net_assets_val,
                    lib_and_netassets_val,
                ) = BS_test(data_context)
                test_dict_acc_val_role.update(
                    {"test_bs__" + c_ref: test_dict_acc_val_t},
                )
                log_kpi_dict_role.update(
                    {
                        "kpi__" + c_ref: {
                            "assets_val": float(assets_val),
                            "liabilities_val": float(liabilities_val),
                            "net_assets_val": float(net_assets_val),
                            "lib_and_netassets_val": float(lib_and_netassets_val),
                        },
                    },
                )
            # PL test
            if ("_StatementOfIncome" in role) or (
                "_ConsolidatedStatementOfIncome" in role
            ):
                test_dict_acc_val_t, pl_val, kpi_val, ordinary_income_val = PL_test(
                    data_context,
                    calc_dict_all[role_suffix],
                )
                test_dict_acc_val_role.update(
                    {"test_pl__" + c_ref: test_dict_acc_val_t},
                )
                log_kpi_dict_role.update(
                    {
                        "kpi__" + c_ref: {
                            "pl_val": float(pl_val),
                            "kpi_val": float(kpi_val),
                            "ordinary_income_val": float(ordinary_income_val),
                        },
                    },
                )
        log_cnt_dict["record_cnt__" + role] = log_cnt_dict_role
        test_dict_acc_val.update({role: test_dict_acc_val_role})
        log_kpi_dict.update({role: log_kpi_dict_role})
        data_list.append(data)
    log_dict["test_dict_acc_val"] = test_dict_acc_val
    log_dict["concat_rate"] = log_concat
    log_dict["log_cnt_dict"] = log_cnt_dict
    log_dict["log_kpi_dict"] = log_kpi_dict

    # save
    out_filename = temp_path_str + "/fs_tbl.pkl"
    FsDataDf(pd.concat(data_list)[get_columns_df(FsDataDf)]).to_pickle(out_filename)
    log_out_filename = temp_path_str + "/fs_tbl_log.json"
    with open(log_out_filename, "w") as f:
        json.dump(log_dict, f, indent=4)
    return 0
    # return FsDataDf(pd.concat(data_list)[get_columns_df(FsDataDf)]),log_dict


def get_zipdir2(docid: str):
    try:
        data_dir_raw = PROJDIR / "data" / "1_raw"
        zip_file = list(data_dir_raw.glob("data_pool_*/" + docid + ".zip"))[0]
    except Exception:
        zip_file = "not found"
    return zip_file


def get_zipdir3(docid: str):
    try:
        data_dir_raw = PROJDIR / "data" / "1_raw"
        zip_file = list(data_dir_raw.glob("data_pool_*/" + docid + ".zip"))[0]
        with ZipFile(str(zip_file)) as zf:
            fn = [
                item for item in zf.namelist() if ("pre.xml" in item) & ("asr" in item)
            ]
            if len(fn) == 0:
                return "no_pre_file"
            return zip_file
    except Exception:
        zip_file = "not_found"
    return zip_file


# %%
from edinet_xbrl_prep.link_base_file_analyzer import get_presentation_account_list


class linkbasefile_for_year:
    def __init__(self, zip_file_str: str, temp_path_str: str):
        self.zip_file_str = zip_file_str
        self.temp_path_str = temp_path_str
        self.log_dict = {}

    def read_linkbase_file(self):
        self.get_presentation_account_list_obj = get_presentation_account_list(
            zip_file_str=self.zip_file_str,
            temp_path_str=self.temp_path_str,
            doc_type="public",
        )
        self.parent_child_df = (
            self.get_presentation_account_list_obj.export_parent_child_link_df()
        )
        self.account_list = (
            self.get_presentation_account_list_obj.export_account_list_df()
        )
        self.log_dict = {
            **self.log_dict,
            **self.get_presentation_account_list_obj.export_log().model_dump(),
        }

    def detect_account_list_year(self):
        head_list = list(
            set(
                self.get_presentation_account_list_obj.export_account_list_df().schima_taxonomi_head,
            ),
        )
        head_jpcrp_list = [
            head
            for head in head_list
            if "http://disclosure.edinet-fsa.go.jp/taxonomy/jpcrp" in head
        ]
        if len(head_jpcrp_list) == 0:
            print("head_jpcrp_list is empty")
            print(head_list)
            return "-"
        head_jpcrp = head_jpcrp_list[0]
        if "2023-12-01" in head_jpcrp:
            self.account_list_year = "2024"
        elif "2022-11-01" in head_jpcrp:
            self.account_list_year = "2023"
        elif "2021-11-01" in head_jpcrp:
            self.account_list_year = "2022"
        elif "2020-11-01" in head_jpcrp:
            self.account_list_year = "2021"
        elif "2019-11-01" in head_jpcrp:
            self.account_list_year = "2020"
        elif "2018-03-31" in head_jpcrp:  # 2019-02-28
            self.account_list_year = "2019"
        elif "2018-02-28" in head_jpcrp:
            self.account_list_year = "2018"
        elif "2017-02-28" in head_jpcrp:
            self.account_list_year = "2017"
        elif "2016-02-29" in head_jpcrp:
            self.account_list_year = "2016"
        elif "2015-03-31" in head_jpcrp:
            self.account_list_year = "2015"
        elif "2013-08-31" in head_jpcrp:
            self.account_list_year = "2014"
        elif "2024-11-01" in head_jpcrp:
            self.account_list_year = "2025"
        else:
            self.account_list_year = "-"
            print("year not found")
        # print("year: ",self.account_list_year)
        return self.account_list_year


def account_taxonomy_year(zip_file_str, temp_path_str):
    if zip_file_str == "not_found" or zip_file_str == "no_pre_file":
        print("data is incomplete")
        return "-"
    linkbasefile_obj = linkbasefile_for_year(
        zip_file_str=zip_file_str,
        temp_path_str=temp_path_str,
    )
    linkbasefile_obj.read_linkbase_file()
    year = linkbasefile_obj.detect_account_list_year()
    return year


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
    fs_dict = {
        "BS": ["_BalanceSheet", "_ConsolidatedBalanceSheet"],
        "PL": ["_StatementOfIncome", "_ConsolidatedStatementOfIncome"],
        "CF": ["_StatementOfCashFlows", "_ConsolidatedStatementOfCashFlows"],
        "SS": [
            "_StatementOfChangesInEquity",
            "_ConsolidatedStatementOfChangesInEquity",
        ],
        "notes": ["_Notes", "_ConsolidatedNotes"],
        "report": ["_CabinetOfficeOrdinanceOnDisclosure"],
    }

    account_list_dir = PROJDIR / "data/0_metadata/common" / "account_list"
    # prep_account_list_common_obj(out_dir=account_list_dir)

    filename = account_list_dir / "account_list_common_obj_dict.pkl"
    with open(filename, "rb") as f:
        account_list_common_obj_dict = pickle.load(f)

    # response_tbl = load_response_tbl()
    n_parallel = 16
    print("max nb process: ", joblib.cpu_count())
    print("max nb process: ", n_parallel)
    out_filename = (
        PROJPATH + "data/3_processed/dataset_2507/BsPlRepoNotes_yuho_250803_chk.pkl.cmp"
    )

    # filename = PROJPATH + "data/3_processed/dataset_2407/response_tbl_with_year.pkl"
    # response_tbl_processed = pd.read_pickle(filename)
    # filename = (
    #    PROJPATH + "data/0_metadata/dataset_2507/response_tbl_rst_2507_v250803.pkl"
    # )
    # response_tbl_latest = pd.read_pickle(filename)
    # response_tbl_unprocessed = response_tbl_latest.query(
    #    "index not in @response_tbl_processed.index",
    # )
    # with tqdm_joblib(
    #    tqdm(desc="My calculation", total=len(response_tbl_unprocessed.index)),
    # ) as progress_bar:
    #    years = joblib.Parallel(n_jobs=n_parallel, verbose=0)(
    #        [
    #            joblib.delayed(account_taxonomy_year)(
    #                zip_file_str=get_zipdir3(itr_docID),
    #                temp_path_str=str(
    #                    PROJDIR
    #                    / "data"
    #                    / "2_intermediate"
    #                    / (
    #                        f"data_pool_{response_tbl_unprocessed.loc[itr_docID, 'dataset']}"
    #                    )
    #                    / itr_docID,
    #                ),
    #            )
    #            for itr_docID in response_tbl_unprocessed.index
    #        ],
    #    )
    # response_tbl_unprocessed["year"] = years
    # print(response_tbl_unprocessed["year"].value_counts())
    # response_tbl_unprocessed.to_pickle(
    #    PROJPATH + "data/3_processed/dataset_2507/response_tbl_with_year_add.pkl",
    # )

    # filename = PROJPATH + "data/3_processed/dataset_2507/response_tbl_with_year_add.pkl"
    # response_tbl = pd.read_pickle(filename)
    # response_tbl = response_tbl.query("year != '-'")
    response_tbl = load_response_tbl()

    if n_parallel == 1:
        results = []
        # logs=[]
        for itr_docID in tqdm(response_tbl.index):
            print(itr_docID)
            result = get_fs_tbl_new(
                # account_list_common_obj_dict=account_list_common_obj_dict,
                docid=itr_docID,
                zip_file_str=get_zipdir2(itr_docID),
                temp_path_str=str(
                    PROJDIR
                    / "data"
                    / "2_intermediate"
                    / (f"data_pool_{response_tbl.loc[itr_docID, 'dataset']}")
                    / itr_docID,
                ),
                role_keyward_list=fs_dict["BS"]
                + fs_dict["PL"]
                + fs_dict["report"]
                + fs_dict["notes"],
            )
            results.append(result)
        joblib.dump(results, out_filename, compress=True)
    else:
        with tqdm_joblib(
            tqdm(desc="My calculation", total=len(response_tbl.index)),
        ) as progress_bar:
            results = joblib.Parallel(n_jobs=n_parallel, verbose=0)(
                [
                    joblib.delayed(get_fs_tbl_new)(
                        account_list_common_obj=account_list_common_obj_dict[
                            response_tbl.loc[itr_docID, "year"]
                        ],
                        docid=itr_docID,
                        zip_file_str=get_zipdir2(itr_docID),
                        temp_path_str=str(
                            PROJDIR
                            / "data"
                            / "2_intermediate"
                            / (f"data_pool_{response_tbl.loc[itr_docID, 'dataset']}")
                            / itr_docID,
                        ),
                        role_keyward_list=fs_dict["BS"]
                        + fs_dict["PL"]
                        + fs_dict["report"]
                        + fs_dict["notes"],
                        update_flg=True,
                    )
                    for itr_docID in response_tbl.index
                ],
            )

        # joblib.dump(results, out_filename, compress=True)
        # joblib.dump(logs, out_filename_log, compress=True)


# %%
if __name__ == "__main__":
    print("start")
    main()
    print("end")

# %% dev
# response_tbl=load_response_tbl()
# fs_tbl_df_list = []
# log_dict_list = []
##response_tbl.head(2).index
# docid_list = ['S100TMNK']#'S100TLFZ'
# for docid in tqdm(docid_list):
#    data_path = response_tbl.loc[docid,'dataset']
#    intermediate_path = PROJDIR / "data" / "2_intermediate" / (f"data_pool_{data_path}") / docid
#    fs_tbl_df, log_dict = get_fs_tbl_new(
#        account_list_common_obj_dict=account_list_common_obj_dict,
#        docid=docid,
#        zip_file_str=get_zipdir2(docid),
#        temp_path_str=intermediate_path,
#        role_keyward_list=fs_dict['BS']+fs_dict['PL']+fs_dict['report']+fs_dict['notes'],
#    )
#    fs_tbl_df_list.append(fs_tbl_df)
#    log_dict_list.append(log_dict)
# fs_tbl_df_all:FsDataDf = FsDataDf(pd.concat(fs_tbl_df_list))


# tmp_tbl_pl = fs_tbl_df_all.query("role == 'rol_ConsolidatedStatementOfIncome'")
# tmp_tbl_bs = fs_tbl_df_all.query("role == 'rol_ConsolidatedBalanceSheet'")
# print(json.dumps(log_dict_list, indent=4))


# %% memo
# filename=PROJPATH+"data/3_processed/dataset_2407/BsPlRepoNotes_yuho_0118.pkl.cmp"
# results=joblib.load(filename)
# results[0][1]
# %%
# filename = PROJPATH+"data/3_processed/dataset_2407/response_tbl_with_year.pkl"
# res = pd.read_pickle(filename)
# res['year'].value_counts()
# %%
def test_get_fs_tbl_new():
    itr_docID = "S100OIW0"
    filename = PROJPATH + "data/3_processed/dataset_2407/response_tbl_with_year.pkl"
    response_tbl = pd.read_pickle(filename)
    response_tbl
    fs_dict = {
        "BS": ["_BalanceSheet", "_ConsolidatedBalanceSheet"],
        "PL": ["_StatementOfIncome", "_ConsolidatedStatementOfIncome"],
        "CF": ["_StatementOfCashFlows", "_ConsolidatedStatementOfCashFlows"],
        "SS": [
            "_StatementOfChangesInEquity",
            "_ConsolidatedStatementOfChangesInEquity",
        ],
        "notes": ["_Notes", "_ConsolidatedNotes"],
        "report": ["_CabinetOfficeOrdinanceOnDisclosure"],
    }
    filename = TESTDIR / "account_list_common_obj_dict.pkl"
    with open(filename, "rb") as f:
        account_list_common_obj_dict = pickle.load(f)
    # itr_docID = 'S100TY62'

    result = get_fs_tbl_new(
        account_list_common_obj=account_list_common_obj_dict[
            response_tbl.loc[itr_docID, "year"]
        ],
        docid=itr_docID,
        zip_file_str=get_zipdir2(itr_docID),
        temp_path_str=str(
            PROJDIR
            / "data"
            / "2_intermediate"
            / (f"data_pool_{response_tbl.loc[itr_docID, 'dataset']}")
            / itr_docID,
        ),
        role_keyward_list=fs_dict["BS"]
        + fs_dict["PL"]
        + fs_dict["report"]
        + fs_dict["notes"],
    )
    temp_path_str = str(
        PROJDIR
        / "data"
        / "2_intermediate"
        / (f"data_pool_{response_tbl.loc[itr_docID, 'dataset']}")
        / itr_docID,
    )
    filename = temp_path_str + "/fs_tbl.pkl"
    log_filename = temp_path_str + "/fs_tbl_log.json"
    log_dict = json.load(open(log_filename))
    fs_data = pd.read_pickle(filename)
    assert len(log_dict) > 0
    assert len(fs_data) > 0


# %%
# test_get_fs_tbl_new()
# ['S1004FWK', 'S1004YZC', 'S10051Q9', 'S1005ALK', 'S1005ALV', 'S1005S2H',
#       'S1005S2J', 'S1005S2R', 'S10079PD', 'S1007VZ7', 'S100AJZW', 'S100CUEY',
#       'S100CUFQ', 'S100CUFZ', 'S100CUG3', 'S100CUG5', 'S100CUG8']
# %%
# %%
