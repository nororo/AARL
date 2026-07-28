# %%
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from tqdm import tqdm

tqdm.pandas()
import difflib
import re
import sys
from functools import partial
from pathlib import Path

sys.path.append(r"/Users/noro/Documents/Projects/t_interpretable_fs")
from src.libs.load_dataset import load_bs_data, load_pl_data

DATADIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/data/1_raw")


PROCDIR = Path("/Users/noro/Documents/Projects/t_interpretable_fs/data/3_processed")
XBRL_PROJPATH = r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
XBRL_PROJDIR = Path(XBRL_PROJPATH)


def get_all_response_tbl():
    filename = XBRL_PROJDIR / "data/3_processed/dataset_2407/response_tbl_with_year.pkl"
    response_tbl = pd.read_pickle(filename)

    filename = (
        XBRL_PROJDIR / "data/3_processed/dataset_2507/response_tbl_with_year_add.pkl"
    )
    response_tbl_add = pd.read_pickle(filename)
    response_tbl_add = response_tbl_add.query("year != '-'")
    response_tbl = pd.concat([response_tbl, response_tbl_add])
    return response_tbl


# %% ############################################################
# split train and test
#############################################################################
# filename = DATADIR / "response_tbl_dataset_train_250929.pkl"
filename = DATADIR / "response_tbl_dataset_train_260218.pkl"
response_train = pd.read_pickle(filename)


x_data = load_bs_data(response_train.index.tolist())
data_pl = load_pl_data(response_train.index.tolist())
# %%
print(response_train.index.nunique())
# %%
print(x_data.docid.nunique())
print(data_pl.docid.nunique())


# %%
x_data.duplicated(subset=["docid", "key"]).sum()
data_pl.duplicated(subset=["docid", "key"]).sum()


# %% check
# %%
def amounts_change_label(diff_rate, threshold_narrow=0.05, threshold_wide=0.2):
    """correct label for consistency to default setting (2026-06-28)"""
    if diff_rate > 2:
        return "new"
    if diff_rate > threshold_wide:
        return "increase_20"
    if diff_rate > threshold_narrow:
        return "increase_5"
    if diff_rate < -threshold_wide:
        return "decrease_20"
    if diff_rate < -threshold_narrow:
        return "decrease_5"
    if (diff_rate >= -threshold_narrow) and (diff_rate <= threshold_narrow):
        return "stable"
    return "missing_value"


# %%
x_data = x_data.assign(
    amounts_change_cls=x_data.diff_rate.apply(amounts_change_label),
)

# %% 0622 当期の総資産に対する比率を追加するか
#fn = "/Users/noro/Documents/Projects/t_interpretable_fs/data/1_raw/merged_bs_filled_unq.pkl"
#bs_data = pd.read_pickle(fn)
#bs_data.columns
#bs_data["data_cy_rate"] = bs_data.data_cy /bs_data.offset
#bs_data["data_cy_rate"].clip(1,-1).hist(bins=50)
# %%
print(
    "label missing rate:",
    len(x_data.query("label_jp_long_filled=='-'")) / len(x_data),
)
# %%
print(
    "calc_parent_key missing rate:",
    len(x_data.query("calc_parent_key == 'not_common_account'")) / len(x_data),
)
# %%

# %%
data_pl = data_pl.assign(
    amounts_change_cls=data_pl.diff_rate.apply(amounts_change_label),
)
# %%
print(len(data_pl.query("label_jp_long_filled.isna()")))
print(len(data_pl.query("docid.isna()")))

print(
    "label missing rate:",
    len(data_pl.query("label_jp_long_filled == '-'")) / len(data_pl),
)
print(
    "calc_parent_key missing rate:",
    len(data_pl.query("calc_parent_key == 'not_common_account'")) / len(data_pl),
)
# %%
df_amounts_change = pd.concat(
    [
        x_data.query("diff_rate.notna() and data_str != '0' and docid.notna()")[
            [
                "key",
                "docid",
                "label_jp_long_filled",
                "diff_rate",
                #"diff_rate_assets",
                "amounts_change_cls",
                "calc_parent_key",
            ]
        ],
        data_pl.query("diff_rate.notna() and data_str != '0' and docid.notna()")[
            [
                "key",
                "docid",
                "label_jp_long_filled",
                "diff_rate",
                #"diff_rate_assets",
                "amounts_change_cls",
                "calc_parent_key",
            ]
        ],
    ],
)
# %%
df_amounts_change.docid.nunique()


# %%
df_amounts_change.to_csv(PROCDIR / "df_amounts_change_train_260218_c.csv")
# %% text
filename = PROCDIR / "text_business_chunked_512_0516_050.csv"
df_text = pd.read_csv(filename)
df_text_train = df_text.query("docid in @response_train.index.tolist()")

filename = PROCDIR / "text_risk_chunked_512_0516_050.csv"
df_text_risk = pd.read_csv(filename)
df_text_risk_train = df_text_risk.query("docid in @response_train.index.tolist()")

filename = PROCDIR / "text_mda_chunked_512_0516_050.csv"
df_text_mda = pd.read_csv(filename)
df_text_mda_train = df_text_mda.query("docid in @response_train.index.tolist()")

# %%
df_text_train = pd.concat([df_text_train, df_text_risk_train, df_text_mda_train])
df_text_train.text_type.value_counts()
# %%
df_text_train.to_csv(PROCDIR / "text_512_train_0516.csv", index=False)

# %%
response_train.index.tolist()
# %% #####################################################################
# test
#############################################################################


# filename = DATADIR / "response_tbl_dataset_test_250929.pkl"
filename = DATADIR / "response_tbl_dataset_eval_260218.pkl"
response_test = pd.read_pickle(filename)
response_test
# %%

x_data = load_bs_data(response_test.index.tolist())
data_pl = load_pl_data(response_test.index.tolist())
# %%

x_data.duplicated(subset=["docid", "key"]).sum()
# data_pl.duplicated(subset=["docid", "key"]).sum()

# %%
x_data = x_data.assign(
    amounts_change_cls=x_data.diff_rate.apply(amounts_change_label),
)

# %%
data_pl = data_pl.assign(
    amounts_change_cls=data_pl.diff_rate.apply(amounts_change_label),
)
# %%
print(len(data_pl.query("label_jp_long_filled.isna()")))
print(len(data_pl.query("docid.isna()")))

# %%
df_amounts_change = pd.concat(
    [
        x_data.query("diff_rate.notna() and data_str != '0' and docid.notna()")[
            [
                "key",
                "docid",
                "label_jp_long_filled",
                "diff_rate",
                #"diff_rate_assets",
                "amounts_change_cls",
                "calc_parent_key",
            ]
        ],
        data_pl.query("diff_rate.notna() and data_str != '0' and docid.notna()")[
            [
                "key",
                "docid",
                "label_jp_long_filled",
                "diff_rate",
                #"diff_rate_assets",
                "amounts_change_cls",
                "calc_parent_key",
            ]
        ],
    ],
)

# %%
df_amounts_change.to_csv(PROCDIR / "df_amounts_change_eval_260218_c.csv")
# %% text
filename = PROCDIR / "text_business_chunked_512_0516_050.csv"
df_text = pd.read_csv(filename)
df_text_test = df_text.query("docid in @response_test.index.tolist()")

filename = PROCDIR / "text_risk_chunked_512_0516_050.csv"
df_text_risk = pd.read_csv(filename)
df_text_risk_test = df_text_risk.query("docid in @response_test.index.tolist()")

filename = PROCDIR / "text_mda_chunked_512_0516_050.csv"
df_text_mda = pd.read_csv(filename)
df_text_mda_test = df_text_mda.query("docid in @response_test.index.tolist()")
print(df_text_mda_test.shape)
# %%
df_text_test = pd.concat([df_text_test, df_text_risk_test, df_text_mda_test])
df_text_test.text_type.value_counts()
# %%
df_text_test.to_csv(PROCDIR / "text_512_eval_0516.csv", index=False)

# %%
response_test.index.tolist()
# %%

df_amounts_change.docid.nunique()
# %%
# %% #####################################################################
# inf
#############################################################################

response_tbl_all = get_all_response_tbl()
# %%
x_data = load_bs_data(response_tbl_all.index.tolist())
data_pl = load_pl_data(response_tbl_all.index.tolist())


# %%
x_data = x_data.assign(
    amounts_change_cls=x_data.diff_rate.apply(amounts_change_label),
)

# %%
data_pl = data_pl.assign(
    amounts_change_cls=data_pl.diff_rate.apply(amounts_change_label),
)
# %%
print(len(data_pl.query("label_jp_long_filled.isna()")))
print(len(data_pl.query("docid.isna()")))

# %%
df_amounts_change = pd.concat(
    [
        x_data.query("diff_rate.notna() and data_str != '0' and docid.notna()")[
            [
                "key",
                "docid",
                "label_jp_long_filled",
                "diff_rate",
                #"diff_rate_assets",
                "amounts_change_cls",
                "calc_parent_key",
            ]
        ],
        data_pl.query("diff_rate.notna() and data_str != '0' and docid.notna()")[
            [
                "key",
                "docid",
                "label_jp_long_filled",
                "diff_rate",
                #"diff_rate_assets",
                "amounts_change_cls",
                "calc_parent_key",
            ]
        ],
    ],
)
# %%
df_amounts_change.docid.nunique()


# %%
df_amounts_change.to_csv(PROCDIR / "df_amounts_change_all_0218_c.csv")

# %%
# %% text
filename = PROCDIR / "text_business_chunked_512_0516_050.csv"
df_text = pd.read_csv(filename)
df_text_all = df_text.query("docid in @response_tbl_all.index.tolist()")

filename = PROCDIR / "text_risk_chunked_512_0516_050.csv"
df_text_risk = pd.read_csv(filename)
df_text_risk_all = df_text_risk.query("docid in @response_tbl_all.index.tolist()")

filename = PROCDIR / "text_mda_chunked_512_0516_050.csv"
df_text_mda = pd.read_csv(filename)
df_text_mda_all = df_text_mda.query("docid in @response_tbl_all.index.tolist()")

# %%
df_text_all = pd.concat([df_text_all, df_text_risk_all, df_text_mda_all])
# %%
df_text_all.to_csv(PROCDIR / "text_512_all_0516.csv", index=False)

# %%
# %% #####################################################################
# inf amd
#############################################################################

# %%

filename = (
    XBRL_PROJDIR
    / "data/3_processed/dataset_2507/restatement/response_tbl_teisei_2507_v260131_with_year.pkl"
)
response_tbl_amd = pd.read_pickle(filename)
response_tbl_amd = response_tbl_amd.query(
    "year != 'not_found' and year != 'no_pre_file'",
)
response_tbl_amd = response_tbl_amd.rename(
    columns={
        "edinetCode": "response_edinetCode",
        "periodStart": "response_periodStart",
        "periodEnd": "response_periodEnd",
        "secCode": "response_secCode",
    },
)
response_tbl_amd.loc["S100X5AB", :]

# %%
from libs.load_dataset import load_bs_data_amd, load_pl_data_amd

x_data = load_bs_data_amd(response_tbl_amd.index.tolist())
data_pl = load_pl_data_amd(response_tbl_amd.index.tolist())

# %%
x_data.query("docid == 'S100X5AB'")  # .loc['S100X5AB',:]
data_pl.query("docid == 'S100X5AB'")
# %%
x_data = x_data.assign(
    amounts_change_cls=x_data.diff_rate.apply(amounts_change_label),
)

# %%
data_pl = data_pl.assign(
    amounts_change_cls=data_pl.diff_rate.apply(amounts_change_label),
)
# %%
print(len(data_pl.query("label_jp_long_filled.isna()")))
print(len(data_pl.query("docid.isna()")))

# %%
df_amounts_change = pd.concat(
    [
        x_data.query("diff_rate.notna() and data_str != '0' and docid.notna()")[
            [
                "key",
                "docid",
                "label_jp_long_filled",
                "diff_rate",
                "amounts_change_cls",
                "calc_parent_key",
            ]
        ],
        data_pl.query("diff_rate.notna() and data_str != '0' and docid.notna()")[
            [
                "key",
                "docid",
                "label_jp_long_filled",
                "diff_rate",
                "amounts_change_cls",
                "calc_parent_key",
            ]
        ],
    ],
)
# %%
df_amounts_change.docid.nunique()


# %%
df_amounts_change.to_csv(PROCDIR / "df_amounts_change_amd_0403_c.csv")

# %%
# %% text
filename = PROCDIR / "text_business_chunked_512_amd_0516_050.csv"
df_text = pd.read_csv(filename)
df_text_all = df_text.query("docid in @response_tbl_amd.index.tolist()")

filename = PROCDIR / "text_risk_chunked_512_amd_0516_050.csv"
df_text_risk = pd.read_csv(filename)
df_text_risk_all = df_text_risk.query("docid in @response_tbl_amd.index.tolist()")

filename = PROCDIR / "text_mda_chunked_512_amd_0516_050.csv"
df_text_mda = pd.read_csv(filename)
df_text_mda_all = df_text_mda.query("docid in @response_tbl_amd.index.tolist()")

# %%
df_text_all = pd.concat([df_text_all, df_text_risk_all, df_text_mda_all])
# %%
df_text_all.to_csv(PROCDIR / "text_512_amd_0516.csv", index=False)
df_text_all.text_type.value_counts()

# %% ####################################################
# reconstruct 1228
#########################################################
filename = PROCDIR / "text_512_train_0516.csv"
df_text_train = pd.read_csv(filename)
df_text_train["row_number"] = df_text_train.index
df_text_train.head()


# %%
def _normalize_chunk_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def _is_redundant_chunk(
    candidate: str,
    accepted: list[str],
    *,
    similarity_threshold: float,
) -> bool:
    """採用済みチャンクと同一または十分に似ていれば True（スキップ対象）。"""
    cand = _normalize_chunk_text(candidate)
    if not cand:
        return True
    for prev in accepted:
        p = _normalize_chunk_text(prev)
        if cand == p:
            return True
        if min(len(cand), len(p)) < 8:
            continue
        if difflib.SequenceMatcher(None, cand, p).ratio() >= similarity_threshold:
            return True
    return False


def _consecutive_group(
    sr: pd.DataFrame,
    start_idx: int,
    max_rows: int = 10,
    max_tokens: int = 512,
    inner_sep: str = "\n",
    *,
    dedupe: bool = True,
    similarity_threshold: float = 0.90,
) -> str:
    """row_number 順に start_idx から走査し、件数・token 上限内でチャンクを採用。

    dedupe が True のとき、正規化後の同一文または類似度が閾値以上のチャンクは
    スキップし、後続行を続けて見る。
    """
    if start_idx >= len(sr) or len(sr) == 0:
        return ""
    texts: list[str] = []
    accepted_for_dedupe: list[str] = []
    tok_sum = 0
    n_taken = 0
    has_tok = "token_size_text" in sr.columns
    for i in range(start_idx, len(sr)):
        if n_taken >= max_rows:
            break
        row = sr.iloc[i]
        raw = str(row["text_list"])
        if dedupe and _is_redundant_chunk(
            raw,
            accepted_for_dedupe,
            similarity_threshold=similarity_threshold,
        ):
            continue
        if has_tok:
            tok = row["token_size_text"]
            tok = 0 if pd.isna(tok) else int(tok)
        else:
            tok = 0
        if n_taken > 0 and tok_sum + tok > max_tokens:
            break
        if n_taken == 0 and tok > max_tokens:
            break
        tok_sum += tok
        texts.append(raw)
        accepted_for_dedupe.append(raw)
        n_taken += 1
    if not texts:
        return ""
    return inner_sep.join(texts)


def _select_text_type_third(
    df_text: pd.DataFrame,
    text_type: str,
    third: int,
    *,
    max_rows: int = 10,
    inner_sep: str = "\n",
) -> str:
    """指定 text_type を row_number 順に並べ、3等分の third（0=前,1=中,2=後）の先頭から連続グループを1つ。

    中区間 [q,2q) の行数が 15 未満なら third=1 は空。後区間 [2q,n) の行数が 10 未満なら third=2 は空。
    """
    df_sub = df_text[df_text["text_type"] == text_type]
    if df_sub.empty:
        return ""
    if "row_number" in df_sub.columns:
        sr = df_sub.sort_values("row_number", kind="mergesort").reset_index(drop=True)
    else:
        sr = df_sub.reset_index(drop=True)
    n = len(sr)
    q = n // 3
    # 3 分割: [0,q), [q,2q), [2q,n) における各区間の行数
    len_mid = min(2 * q, n) - q
    len_back = max(0, n - 2 * q)
    if third == 1 and len_mid < max_rows * 2:
        return ""
    if third == 2 and len_back < max_rows:
        return ""
    start_idx = min(third * q, n - 1)
    seg = _consecutive_group(sr, start_idx, inner_sep=inner_sep, max_rows=max_rows)
    return (seg + "\n") if seg else ""


def _select_text_mda_third(df_text: pd.DataFrame, third: int) -> str:
    return _select_text_type_third(df_text, "mda", third, max_rows=5)


def _select_text_business_third(df_text: pd.DataFrame, third: int) -> str:
    return _select_text_type_third(df_text, "business", third, max_rows=5)


def _select_text_risk_third(df_text: pd.DataFrame, third: int) -> str:
    return _select_text_type_third(df_text, "risk", third, max_rows=5)


def select_text_mix(df_text: pd.DataFrame):
    parts: list[str] = []
    b = _select_text_type_third(df_text, "business", 0, max_rows=5)
    if b:
        parts.append(b.rstrip("\n"))
    r = _select_text_type_third(df_text, "risk", 0, max_rows=5)
    if r:
        parts.append(r.rstrip("\n"))
    m = _select_text_mda_third(df_text, 0)
    if m:
        parts.append(m.rstrip("\n"))
    return ("\n".join(parts) + "\n") if parts else ""


# %%
df_text_train_reconstructed_mda_1 = (
    df_text_train.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=0))
    .reset_index(name="text_list")
)

df_text_train_reconstructed_mda_2 = (
    df_text_train.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=1))
    .reset_index(name="text_list")
)
df_text_train_reconstructed_mda_3 = (
    df_text_train.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=2))
    .reset_index(name="text_list")
)

df_text_train_reconstructed_business_1 = (
    df_text_train.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=0))
    .reset_index(name="text_list")
)
df_text_train_reconstructed_business_2 = (
    df_text_train.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=1))
    .reset_index(name="text_list")
)
df_text_train_reconstructed_business_3 = (
    df_text_train.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=2))
    .reset_index(name="text_list")
)

df_text_train_reconstructed_risk_1 = (
    df_text_train.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=0))
    .reset_index(name="text_list")
)
df_text_train_reconstructed_risk_2 = (
    df_text_train.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=1))
    .reset_index(name="text_list")
)
df_text_train_reconstructed_risk_3 = (
    df_text_train.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=2))
    .reset_index(name="text_list")
)

# %%
df_text_train_reconstructed = pd.concat(
    [
        # df_text_train.query('text_type not in ["mda", "business", "risk"]'),
        df_text_train_reconstructed_mda_1.assign(text_type="mda"),
        df_text_train_reconstructed_mda_2.assign(text_type="mda"),
        df_text_train_reconstructed_mda_3.assign(text_type="mda"),
        df_text_train_reconstructed_business_1.assign(text_type="business"),
        df_text_train_reconstructed_business_2.assign(text_type="business"),
        df_text_train_reconstructed_business_3.assign(text_type="business"),
        df_text_train_reconstructed_risk_1.assign(text_type="risk"),
        df_text_train_reconstructed_risk_2.assign(text_type="risk"),
        df_text_train_reconstructed_risk_3.assign(text_type="risk"),
    ],
)[["docid", "text_type", "text_list"]]
# %%
df_text_train_reconstructed.query("text_list != ''").to_csv(
    PROCDIR / "text_512_train_reconstructed_0516.csv",
    index=False,
)
# %%

df_text_train_reconstructed.query(
    "text_list != ''",
).text_type.value_counts()  # .text_type.value_counts()
# %%
print(df_text_train_reconstructed.iloc[283245]["text_type"])
print(df_text_train_reconstructed.iloc[283245]["text_list"])


# %%
filename = PROCDIR / "text_512_eval_0516.csv"
df_text_test = pd.read_csv(filename)
df_text_test

# %%
df_text_test_reconstructed_smp_1 = (
    df_text_test.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=0))
    .reset_index(name="text_list")
)
df_text_test_reconstructed_smp_2 = (
    df_text_test.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=1))
    .reset_index(name="text_list")
)
df_text_test_reconstructed_smp_3 = (
    df_text_test.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=2))
    .reset_index(name="text_list")
)

df_text_test_reconstructed_business_1 = (
    df_text_test.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=0))
    .reset_index(name="text_list")
)
df_text_test_reconstructed_business_2 = (
    df_text_test.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=1))
    .reset_index(name="text_list")
)
df_text_test_reconstructed_business_3 = (
    df_text_test.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=2))
    .reset_index(name="text_list")
)
df_text_test_reconstructed_risk_1 = (
    df_text_test.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=0))
    .reset_index(name="text_list")
)
df_text_test_reconstructed_risk_2 = (
    df_text_test.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=1))
    .reset_index(name="text_list")
)
df_text_test_reconstructed_risk_3 = (
    df_text_test.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=2))
    .reset_index(name="text_list")
)
# %%
df_text_test_reconstructed = pd.concat(
    [
        # df_text_test.query('text_type not in ["mda", "business", "risk"]'),
        df_text_test_reconstructed_smp_1.assign(text_type="mda"),
        df_text_test_reconstructed_smp_2.assign(text_type="mda"),
        df_text_test_reconstructed_smp_3.assign(text_type="mda"),
        df_text_test_reconstructed_business_1.assign(text_type="business"),
        df_text_test_reconstructed_business_2.assign(text_type="business"),
        df_text_test_reconstructed_business_3.assign(text_type="business"),
        df_text_test_reconstructed_risk_1.assign(text_type="risk"),
        df_text_test_reconstructed_risk_2.assign(text_type="risk"),
        df_text_test_reconstructed_risk_3.assign(text_type="risk"),
    ],
)[["docid", "text_type", "text_list"]]
# %%
df_text_test_reconstructed.query("text_list != ''").to_csv(
    PROCDIR / "text_512_eval_reconstructed_0516.csv",
    index=False,
)
# %% amd
filename = PROCDIR / "text_512_amd_0516.csv"
df_text_amd = pd.read_csv(filename)
df_text_amd.text_type.value_counts()

# %%

df_text_amd_reconstructed_smp_1 = (
    df_text_amd.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=0))
    .reset_index(name="text_list")
)
df_text_amd_reconstructed_smp_2 = (
    df_text_amd.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=1))
    .reset_index(name="text_list")
)
df_text_amd_reconstructed_smp_3 = (
    df_text_amd.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=2))
    .reset_index(name="text_list")
)

df_text_amd_reconstructed_business_1 = (
    df_text_amd.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=0))
    .reset_index(name="text_list")
)
df_text_amd_reconstructed_business_2 = (
    df_text_amd.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=1))
    .reset_index(name="text_list")
)
df_text_amd_reconstructed_business_3 = (
    df_text_amd.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=2))
    .reset_index(name="text_list")
)
df_text_amd_reconstructed_risk_1 = (
    df_text_amd.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=0))
    .reset_index(name="text_list")
)
df_text_amd_reconstructed_risk_2 = (
    df_text_amd.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=1))
    .reset_index(name="text_list")
)
df_text_amd_reconstructed_risk_3 = (
    df_text_amd.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=2))
    .reset_index(name="text_list")
)
# %%
df_text_amd_reconstructed = pd.concat(
    [
        # df_text_amd.query('text_type not in ["mda", "business", "risk"]'),
        df_text_amd_reconstructed_smp_1.assign(text_type="mda"),
        df_text_amd_reconstructed_smp_2.assign(text_type="mda"),
        df_text_amd_reconstructed_smp_3.assign(text_type="mda"),
        df_text_amd_reconstructed_business_1.assign(text_type="business"),
        df_text_amd_reconstructed_business_2.assign(text_type="business"),
        df_text_amd_reconstructed_business_3.assign(text_type="business"),
        df_text_amd_reconstructed_risk_1.assign(text_type="risk"),
        df_text_amd_reconstructed_risk_2.assign(text_type="risk"),
        df_text_amd_reconstructed_risk_3.assign(text_type="risk"),
    ],
)[["docid", "text_type", "text_list"]]

# %%
df_text_amd_reconstructed.query("text_list != ''").to_csv(
    PROCDIR / "text_512_amd_reconstructed_0516.csv",
    index=False,
)

# %%
df_text_amd_reconstructed.text_type.value_counts()
# %% all

filename = PROCDIR / "text_512_all_0516.csv"
df_text_all = pd.read_csv(filename)
df_text_all


# %%

df_text_all_reconstructed_smp_1 = (
    df_text_all.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=0))
    .reset_index(name="text_list")
)
df_text_all_reconstructed_smp_2 = (
    df_text_all.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=1))
    .reset_index(name="text_list")
)
df_text_all_reconstructed_smp_3 = (
    df_text_all.query("text_type == 'mda'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_mda_third, third=2))
    .reset_index(name="text_list")
)

df_text_all_reconstructed_business_1 = (
    df_text_all.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=0))
    .reset_index(name="text_list")
)
df_text_all_reconstructed_business_2 = (
    df_text_all.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=1))
    .reset_index(name="text_list")
)
df_text_all_reconstructed_business_3 = (
    df_text_all.query("text_type == 'business'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_business_third, third=2))
    .reset_index(name="text_list")
)
df_text_all_reconstructed_risk_1 = (
    df_text_all.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=0))
    .reset_index(name="text_list")
)
df_text_all_reconstructed_risk_2 = (
    df_text_all.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=1))
    .reset_index(name="text_list")
)
df_text_all_reconstructed_risk_3 = (
    df_text_all.query("text_type == 'risk'")
    .groupby(
        "docid",
    )
    .progress_apply(partial(_select_text_risk_third, third=2))
    .reset_index(name="text_list")
)
# %%
df_text_all_reconstructed = pd.concat(
    [
        # df_text_all.query('text_type not in ["mda", "business", "risk"]'),
        df_text_all_reconstructed_smp_1.assign(text_type="mda"),
        df_text_all_reconstructed_smp_2.assign(text_type="mda"),
        df_text_all_reconstructed_smp_3.assign(text_type="mda"),
        df_text_all_reconstructed_business_1.assign(text_type="business"),
        df_text_all_reconstructed_business_2.assign(text_type="business"),
        df_text_all_reconstructed_business_3.assign(text_type="business"),
        df_text_all_reconstructed_risk_1.assign(text_type="risk"),
        df_text_all_reconstructed_risk_2.assign(text_type="risk"),
        df_text_all_reconstructed_risk_3.assign(text_type="risk"),
    ],
)[["docid", "text_type", "text_list"]]

# %%
df_text_all_reconstructed.to_csv(
    PROCDIR / "text_512_all_reconstructed_0516.csv",
    index=False,
)
# %%
df_text_all_reconstructed.text_type.value_counts()

# %%
from datetime import timedelta


def add_nextdoc(df):
    metadata_docid = df.copy()  # .set_index('docID')
    metadata_docid["periodStart"] = pd.to_datetime(
        df["response_periodStart"],
    )  # .map(fnc_date_formatted)
    metadata_docid["periodEnd"] = pd.to_datetime(
        df["response_periodEnd"],
    )  # .map(fnc_date_formatted)

    metadata_docid["periodEnd_plus_1month"] = pd.to_datetime(
        df["response_periodEnd"],
    ) + timedelta(days=1)
    # metadata_docid['company_id']=df.response_edinetCode

    metadata_docid["pre_key"] = (
        metadata_docid.response_edinetCode
        + "_"
        + metadata_docid["periodEnd_plus_1month"].dt.strftime("%Y%m%d")
    )
    metadata_docid["cur_key"] = (
        metadata_docid.response_edinetCode
        + "_"
        + metadata_docid["periodStart"].dt.strftime("%Y%m%d")
    )

    metadata_docid = (
        pd.merge(
            metadata_docid.reset_index(),
            metadata_docid.reset_index()[["docID", "cur_key"]].rename(
                columns={"docID": "docID_next"},
            ),
            left_on="pre_key",
            right_on="cur_key",
            how="left",
        )
        .drop(["cur_key_y", "pre_key", "cur_key_x", "periodEnd_plus_1month"], axis=1)
        .set_index("docID")
    )
    return metadata_docid  # [['docID_next','periodStart','periodEnd']]


filename = DATADIR / "response_tbl_dataset_train_260218.pkl"
response_train = pd.read_pickle(filename)

docID_next_train = add_nextdoc(response_train)  # ["docID_next"]
docID_next_train.to_pickle(DATADIR / "response_tbl_dataset_train_260218_next.pkl")

# %% 0427
# check amd
filename = DATADIR / "response_tbl_dataset_eval_260218.pkl"
response_amd = pd.read_pickle(filename)

# %%
df_amounts_change = pd.read_csv(PROCDIR / "df_amounts_change_amd_0403.csv")
df_amounts_change.docid.nunique()
df_amounts_change.query("docid == 'S100X5AB'")
# %%
fn = "/Users/noro/Documents/Projects/t_interpretable_fs/data/2_intermediate/feature/fs_pca_comp512/fsdata_dim_reduced_pca_fraud.pkl"
df_fs = pd.read_pickle(fn)
df_fs.query("index == 'S100X5AB'")
# %%
fn = "/Users/noro/Documents/Projects/t_interpretable_fs/data/2_intermediate/feature/text_lda_topics256_maxf2048/text_lda_amd.pkl"
df_text = pd.read_pickle(fn)
df_text.query("index == 'S100X5AB'")
# %%

# %%
df_text_risk.query("docid == 'S100X5AB'")
# %%
