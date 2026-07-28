# %%
import sys

import pandas as pd

sys.path.append(r"/Users/noro/Documents/Projects/XBRL_common_space_projection")
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


import sys

sys.path.append(r"/Users/noro/Documents/Projects/XBRL_common_space_projection")
sys.path.append(
    r"/Users/noro/Documents/Projects/XBRL_common_space_projection/src/edinet_xbrl_prep",
)


from edinet_xbrl_prep.link_base_file_analyzer import *

# %%
PROJPATH = r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR = Path(PROJPATH)
TESTDIR = Path(PROJPATH) / "tests/20250127"
TESTDIR


class bs_pl_table:
    """type
    all columns other than 'depth': string
    depth: float
    """

    def __init__(self):
        self.f_name_dict = {
            2014: "2014account_list.xls",
            2015: "2015account_list.xls",
            2016: "2016account_list.xls",
            2017: "2017account_list.xls",
            2018: "1f_AccountList_2018.xls",
            2019: "1f_AccountList_2019.xls",
            2020: "1f_AccountList_2020.xlsx",
            2021: "1f_AccountList_2021.xlsx",
            2022: "1f_AccountList_2022.xlsx",
            2023: "1f_AccountList_2023.xlsx",
            2024: "1f_AccountList_2024.xlsx",
            2025: "1f_AccountList_2025.xlsx",
        }
        self.rename_dict = {
            "科目分類": "account_type",
            "標準ラベル（日本語）": "label_jp",
            "冗長ラベル（日本語）": "label_jp_long",
            "標準ラベル（英語）": "label_en",
            "冗長ラベル（英語）": "label_en_long",
            "用途区分、財務諸表区分及び業種区分のラベル（日本語）": "label_jp_purpose",
            "用途区分、財務諸表区分及び業種区分のラベル（英語）": "label_en_purpose",
            "名前空間プレフィックス": "namespace_prefix",
            "要素名": "element_name",
        }
        self.out_columns = [
            "account_type",
            "label_jp",
            "label_jp_long",
            "label_en",
            "label_en_long",
            "label_jp_purpose",
            "label_en_purpose",
            "namespace_prefix",
            "element_name",
            #'type',
            #'substitutionGroup',
            "periodType",
            "balance",
            "abstract",
            "depth",
            "key_cap",
            "parent_key",
            "key",
            "schima",
            "taxonomi",
            "is_parent_abstruct",
            "sum_flg",
            "bussiness_type_num_str",
            "year_str",
            "tbl_name",
            #'参照リンク'
        ]

        #        self.out_columns=['科目分類', '標準ラベル（日本語）', '冗長ラベル（日本語）', '標準ラベル（英語）', '冗長ラベル（英語）',
        #            '用途区分、財務諸表区分及び業種区分のラベル（日本語）', '用途区分、財務諸表区分及び業種区分のラベル（英語）',
        #            '名前空間プレフィックス', '要素名', 'type', 'substitutionGroup', 'periodType',
        #            'balance', 'abstract', 'depth', 'key_cap', 'parent_key','key','schima','taxonomi',
        #            'is_parent_abstruct', 'sum_flg', 'bussiness_type_num_str', 'year_str','tbl_name',
        #            #'参照リンク'
        #            ]
        self.integlated_list_path = Path(
            "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/2_intermediate/account_list",
        )
        if self.integlated_list_path.is_dir():
            self.tbl_bs_c = self.post_proc(
                pd.read_csv(
                    str(self.integlated_list_path / "account_list_bs.csv"),
                    encoding="shift-jis",
                    dtype=str,
                    index_col=None,
                ),
            )
            self.tbl_bs_c.depth = self.tbl_bs_c.depth.astype(float)

            self.tbl_pl_c = self.post_proc(
                pd.read_csv(
                    str(self.integlated_list_path / "account_list_pl.csv"),
                    encoding="shift-jis",
                    dtype=str,
                    index_col=None,
                ),
            )
            self.tbl_pl_c.depth = self.tbl_pl_c.depth.astype(float)
            self.tbl_oci_c = self.post_proc(
                pd.read_csv(
                    str(self.integlated_list_path / "account_list_oci.csv"),
                    encoding="shift-jis",
                    dtype=str,
                    index_col=None,
                ),
            )
            self.tbl_oci_c.depth = self.tbl_oci_c.depth.astype(float)
            self.tbl_ss_c = self.post_proc(
                pd.read_csv(
                    str(self.integlated_list_path / "account_list_ss.csv"),
                    encoding="shift-jis",
                    dtype=str,
                    index_col=None,
                ),
            )
            self.tbl_ss_c.depth = self.tbl_ss_c.depth.astype(float)
            self.tbl_cf_c = self.post_proc(
                pd.read_csv(
                    str(self.integlated_list_path / "account_list_cf.csv"),
                    encoding="shift-jis",
                    dtype=str,
                    index_col=None,
                ),
            )
            self.tbl_cf_c.depth = self.tbl_cf_c.depth.astype(float)
            self.load = True
        else:
            self.load = False

    def post_proc(self, tbl):
        tbl.depth = tbl.depth.astype(float)
        tbl.balance = tbl.balance.fillna("-")
        return tbl.rename(columns=self.rename_dict)

    def get_bs_pl_all(self):
        if self.load:
            return (
                self.tbl_bs_c[self.out_columns],
                self.tbl_pl_c[self.out_columns],
                self.tbl_oci_c[self.out_columns],
                self.tbl_ss_c[self.out_columns],
                self.tbl_cf_c[self.out_columns],
            )
        return self.make_bs_pl_all()

    def make_bs_pl_all(self):
        # filename="/Users/noro/Documents/Projects/XBRLanalysis/data/bkup/metadata/"+str(year)+"account_list.xls"
        tbl_bs_c = pd.DataFrame()
        tbl_pl_c = pd.DataFrame()
        tbl_oci_c = pd.DataFrame()
        tbl_ss_c = pd.DataFrame()
        tbl_cf_c = pd.DataFrame()

        for year in self.f_name_dict.keys():
            filename = (
                "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/0_metadata/xbrl_keys/"
                + self.f_name_dict[year]
            )
            print(filename)
            # filename="/Users/noro/Documents/Projects/XBRLanalysis/data/bkup/metadata/"+self.f_name_dict[year]
            if year >= 2020:
                book = pd.ExcelFile(filename, engine="openpyxl")
            else:
                book = pd.ExcelFile(filename)

            for itr in range(2, len(book.sheet_names) - 1):
                # self.itr=itr
                # self.fname=self.f_name_dict[year]
                sheet_name = itr
                if itr == 2:
                    tbl_bs, tbl_pl, tbl_oci, tbl_ss, tbl_cf = self.get_bs_pl(
                        book,
                        sheet_name,
                        year,
                    )
                    tbl_bs_c = pd.concat([tbl_bs_c, tbl_bs], axis=0)
                    tbl_pl_c = pd.concat([tbl_pl_c, tbl_pl], axis=0)
                    tbl_oci_c = pd.concat([tbl_oci_c, tbl_oci], axis=0)
                    tbl_ss_c = pd.concat([tbl_ss_c, tbl_ss], axis=0)
                    tbl_cf_c = pd.concat([tbl_cf_c, tbl_cf], axis=0)

                else:
                    tbl_bs, tbl_pl, tbl_ss, tbl_cf = self.get_bs_pl(
                        book,
                        sheet_name,
                        year,
                    )
                    tbl_bs2, tbl_pl2, tbl_oci, tbl_ss2, tbl_cf2 = self.get_bs_pl(
                        book,
                        2,
                        year,
                    )
                    tbl_bs = self.merge_general_mst(tbl_bs, tbl_bs2)
                    tbl_pl = self.merge_general_mst(tbl_pl, tbl_pl2)
                    tbl_ss = self.merge_general_mst(tbl_ss, tbl_ss2)
                    # tbl_oci=self.merge_general_mst(tbl_oci,tbl_oci2)
                    tbl_cf = self.merge_general_mst(tbl_cf, tbl_cf2)

                    tbl_bs_c = pd.concat([tbl_bs_c, tbl_bs], axis=0)
                    tbl_pl_c = pd.concat([tbl_pl_c, tbl_pl], axis=0)
                    tbl_ss_c = pd.concat([tbl_ss_c, tbl_ss], axis=0)
                    tbl_cf_c = pd.concat([tbl_cf_c, tbl_cf], axis=0)

                self.tbl_bs_c = tbl_bs_c[self.out_columns].rename(
                    columns=self.rename_dict,
                )
                self.tbl_pl_c = tbl_pl_c[self.out_columns].rename(
                    columns=self.rename_dict,
                )
                self.tbl_oci_c = tbl_oci_c[self.out_columns].rename(
                    columns=self.rename_dict,
                )
                self.tbl_ss_c = tbl_ss_c[self.out_columns].rename(
                    columns=self.rename_dict,
                )
                self.tbl_cf_c = tbl_cf_c[self.out_columns].rename(
                    columns=self.rename_dict,
                )
                self.save_bs_pl()
        return (
            self.tbl_bs_c[self.out_columns],
            self.tbl_pl_c[self.out_columns],
            self.tbl_oci_c[self.out_columns],
            self.tbl_ss_c[self.out_columns],
            self.tbl_cf_c[self.out_columns],
        )

    def save_bs_pl(self):
        self.integlated_list_path.mkdir(parents=True, exist_ok=True)
        self.tbl_bs_c.to_csv(
            str(self.integlated_list_path / "account_list_bs.csv"),
            encoding="shift-jis",
        )
        self.tbl_pl_c.to_csv(
            str(self.integlated_list_path / "account_list_pl.csv"),
            encoding="shift-jis",
        )
        self.tbl_oci_c.to_csv(
            str(self.integlated_list_path / "account_list_oci.csv"),
            encoding="shift-jis",
        )
        self.tbl_ss_c.to_csv(
            str(self.integlated_list_path / "account_list_ss.csv"),
            encoding="shift-jis",
        )
        self.tbl_cf_c.to_csv(
            str(self.integlated_list_path / "account_list_cf.csv"),
            encoding="shift-jis",
        )

    def get_bs_pl(self, book, sheet_name, year):
        sheet = book.parse(sheet_name=sheet_name, header=1, index_col=0).reset_index()
        mask = sheet["科目分類"].str.extract("(.+科目一覧)").notna()
        startpoint = np.where(mask)[0]
        tbl_bs = sheet.iloc[: startpoint[0] - 1].reset_index(drop=True)
        tbl_bs = self._prep(tbl_bs, year, sheet_name)
        tbl_bs = tbl_bs.assign(tbl_name="BS")

        tbl_pl = sheet.iloc[startpoint[0] + 2 : startpoint[1] - 1].reset_index(
            drop=True,
        )
        tbl_pl = self._prep(tbl_pl, year, sheet_name)
        tbl_pl = tbl_pl.assign(tbl_name="PL")

        if sheet_name == 2:
            tbl_oci = sheet.iloc[startpoint[1] + 2 : startpoint[2] - 1].reset_index(
                drop=True,
            )
            tbl_oci = self._prep(tbl_oci, year, sheet_name)
            tbl_oci = tbl_oci.assign(tbl_name="OCI")
            tbl_ss = sheet.iloc[startpoint[2] + 2 : startpoint[3] - 1].reset_index(
                drop=True,
            )
            tbl_ss = self._prep(tbl_ss, year, sheet_name)
            tbl_ss = tbl_ss.assign(tbl_name="SS")
            tbl_cf = sheet.iloc[startpoint[3] + 2 :].reset_index(drop=True)
            mask_cf = tbl_cf["科目分類"].notna()
            tbl_cf = tbl_cf.loc[mask_cf, :]
            tbl_cf = self._prep(tbl_cf, year, sheet_name)
            tbl_cf = tbl_cf.assign(tbl_name="CF")

            return tbl_bs, tbl_pl, tbl_oci, tbl_ss, tbl_cf
        tbl_ss = sheet.iloc[startpoint[1] + 2 : startpoint[2] - 1].reset_index(
            drop=True,
        )
        tbl_ss = self._prep(tbl_ss, year, sheet_name)
        tbl_ss = tbl_ss.assign(tbl_name="SS")
        tbl_cf = sheet.iloc[startpoint[2] + 2 :].reset_index(drop=True)
        mask_cf = tbl_cf["科目分類"].notna()
        tbl_cf = tbl_cf.loc[mask_cf, :]
        tbl_cf = self._prep(tbl_cf, year, sheet_name)
        tbl_cf = tbl_cf.assign(tbl_name="CF")
        return tbl_bs, tbl_pl, tbl_ss, tbl_cf

    def _prep(self, tbl, year, sheet_name):
        tbl = tbl.assign(key_cap=tbl["名前空間プレフィックス"] + ":" + tbl["要素名"])
        tbl = tbl.assign(
            key=tbl.key_cap.str.lower(),
            parent_key=["no_parent_key"]
            + [self._get_parent_key(tbl, itr) for itr in range(1, len(tbl))],
            is_parent_abstruct=[True]
            + [self._is_parent_abstruct(tbl, itr) for itr in range(1, len(tbl))],
            # 日本語のカラムだと、ラベル名称に「合計」が入ってる場合がある。
            sum_flg=(
                (
                    tbl["用途区分、財務諸表区分及び業種区分のラベル（英語）"]
                    .str.extract("(.+合計)")
                    .notna()
                )
                * 1
            )
            .astype(int)
            .astype(str),
            bussiness_type_num_str=str(sheet_name),
            year_str=str(year),
        )
        tbl["schima"] = tbl.key.str.split(":", expand=True)[0]
        tbl["taxonomi"] = (tbl.key.str.split(":", expand=True)[1]).fillna("-")
        tbl.depth = tbl.depth.astype(float)
        return tbl

    def _get_parent_key(self, tbl_bs, itr):
        # self.tbl_bs=tbl_bs
        # self.itr=itr
        serch_obj = tbl_bs.depth.iloc[itr] - 1
        rcd_obj = tbl_bs.query("depth==@serch_obj and index < @itr").tail(1)

        return rcd_obj.key_cap.values[0]

    def _is_parent_abstruct(self, tbl_bs, itr):
        serch_obj = tbl_bs.depth.iloc[itr] - 1
        rcd_obj = tbl_bs.query("depth==@serch_obj and index < @itr").tail(1)
        return rcd_obj.abstract.values[0] == "true"

    def merge_general_mst(self, tbl_bs, tbl_bs2):
        # TODO otherwize than BNK should be considered
        # if tbl_bs.bussiness_type_num_str_base.max()=='4':
        tbl_bs = tbl_bs.assign(
            key_cap_normalized=tbl_bs.key_cap
            # .str.removesuffix('AssetsBNK')
            # .str.removesuffix('LiabilitiesBNK')
            # .str.removesuffix('OIBNK')
            .str.removesuffix("BNK")
            .str.removesuffix("CNS")
            .str.removesuffix("CNA")
            .str.removesuffix("SEC")
            .str.removesuffix("INS")
            .str.removesuffix("RWY")
            .str.removesuffix("WAT")
            .str.removesuffix("NWY")
            .str.removesuffix("telecommunications")
            .str.removesuffix("ELE")
            .str.removesuffix("GAS")
            .str.removesuffix("LIQ")
            .str.removesuffix("IVT")
            .str.removesuffix("INV")
            .str.removesuffix("SPF")
            .str.removesuffix("MED")
            .str.removesuffix("EDU")
            .str.removesuffix("CMD")
            .str.removesuffix("LEA")
            .str.removesuffix("FND"),
        )
        # tbl_bs2=tbl_bs2.assign(key_cap_normalized=tbl_bs2.key_cap
        #                       .str.removesuffix('CA')
        #                       .str.removesuffix('OA')
        #                       )
        tbl_bs2 = tbl_bs2.assign(
            match_flg=(tbl_bs2.key_cap.isin(tbl_bs.key_cap_normalized)).astype(int),
        )
        tbl_bs2_match = pd.merge(
            tbl_bs2.query("match_flg==1"),
            tbl_bs[
                [
                    "key_cap_normalized",
                    "depth",
                    "parent_key",
                    "is_parent_abstruct",
                    "bussiness_type_num_str",
                ]
            ].rename(
                columns={
                    "depth": "depth_base",
                    "parent_key": "parent_key_base",
                    "is_parent_abstruct": "is_parent_abstruct_base",
                    "bussiness_type_num_str": "bussiness_type_num_str_base",
                },
            ),
            left_on="key_cap",
            right_on="key_cap_normalized",
            how="left",
        )
        # matchした場合、depthとparentを変更
        tbl_bs2_match = tbl_bs2_match.assign(
            depth=tbl_bs2_match.depth_base,
            parent_key=tbl_bs2_match.parent_key_base,
            is_parent_abstruct=tbl_bs2_match.is_parent_abstruct_base,
            key_cap=tbl_bs2_match.key_cap,
            bussiness_type_num_str=tbl_bs2_match.bussiness_type_num_str_base,
        )
        # unmatchedのaccountのdepthをparentに基づいて変更 depth++1の繰り返し処理
        max_depth = int(tbl_bs2.depth.max())
        for itr_depth in range(2, max_depth + 1):
            tbl_bs2_match_add = pd.merge(
                tbl_bs2.query("depth==@itr_depth and match_flg==0"),
                tbl_bs2_match[
                    ["key_cap_normalized", "depth_base", "bussiness_type_num_str_base"]
                ],
                left_on="parent_key",
                right_on="key_cap_normalized",
                how="left",
            )
            tbl_bs2_match_add = tbl_bs2_match_add.assign(
                depth=tbl_bs2_match_add.depth_base + 1,
                bussiness_type_num_str=tbl_bs2_match_add.bussiness_type_num_str_base,
            )
            tbl_bs2_match_add = tbl_bs2_match_add.assign(
                depth_base=tbl_bs2_match_add.depth,
                key_cap_normalized=tbl_bs2_match_add.key_cap,
            )
            tbl_bs2_match = pd.concat([tbl_bs2_match, tbl_bs2_match_add], axis=0)
        return pd.concat(
            [tbl_bs, tbl_bs2_match[self.out_columns]],
            axis=0,
        ).drop_duplicates(keep="last", subset="key_cap")


def g_argmax(x):
    return x.value_counts().idxmax()


# %%
bs_pl_table_loader = bs_pl_table()
bs_temp, pl_temp, oci_temp, ss_temp, cf_temp = bs_pl_table_loader.get_bs_pl_all()


# %% BS 補完用データ
def get_cy_bs_for_complemetion():
    filename = PROJDIR / "data/3_processed/dataset_2507" / "cy_bs.pkl"
    cy_bs = pd.read_pickle(filename)

    # calcによって計算(定義されている場合)
    cy_bs["sign_lab"] = (cy_bs.debit_flg - cy_bs.credit_flg).replace(
        {1: "debit", -1: "credit", 0: None},
    )
    cy_bs["sign_lab"].isna().sum()

    def g_argmax(x):
        return x.value_counts().idxmax()

    sign_lab_dict = (
        cy_bs[["sign_lab", "key"]]
        .query("sign_lab.notna()")
        .groupby("key")
        .agg({"sign_lab": g_argmax})
    )
    print(len(sign_lab_dict))
    cy_bs["sign_lab"] = cy_bs["sign_lab"].fillna(
        cy_bs["key"].map(sign_lab_dict["sign_lab"]),
    )
    cy_bs["sign_lab"].value_counts()
    cy_bs["sign_lab"].isna().sum()
    return cy_bs, sign_lab_dict


# %% main BS
filename = PROJDIR / "data/3_processed/dataset_2507" / "merged_bs.pkl"
merged_bs = pd.read_pickle(filename)
merged_bs["sign_lab"] = (merged_bs.debit_flg - merged_bs.credit_flg).replace(
    {1: "debit", -1: "credit", 0: None},
)

cy_bs, sign_lab_dict = get_cy_bs_for_complemetion()

merged_bs["sign_lab"] = merged_bs["sign_lab"].fillna(
    merged_bs["key"].map(sign_lab_dict["sign_lab"]),
)
merged_bs.head()

# fillna_1 (Excelで補う)
bs_balance_df = bs_temp[["balance", "key_cap"]].drop_duplicates().set_index("key_cap")
print("number of missing sign_lab", merged_bs.sign_lab.isna().sum())
merged_bs["sign_lab"] = merged_bs["sign_lab"].fillna(
    merged_bs["key"].map(bs_balance_df["balance"]),
)

# fillna_2 (他の企業のlabelで欠損を補完)

labname_dict = (
    cy_bs[["label_jp", "key"]]
    .query("label_jp.notna()")
    .groupby("key")
    .agg({"label_jp": g_argmax})
)
merged_bs["label_jp_filled"] = merged_bs["label_jp"].fillna(
    merged_bs["key"].map(labname_dict["label_jp"]),
)
labname_en_dict = (
    cy_bs[["label_en", "key"]]
    .query("label_en.notna() and label_en != '-'")
    .groupby("key")
    .agg({"label_en": g_argmax})
)
label_en_filled = (
    merged_bs["label_en"]
    .replace({"-": None})
    .fillna(merged_bs["key"].map(labname_en_dict["label_en"]))
)
# %%
print(merged_bs["sign_lab"].value_counts())
# %%
merged_bs["label_jp_long_filled"] = merged_bs.label_jp_long.replace({"-": None}).fillna(
    merged_bs.label_jp_filled,
)  # .value_counts()
merged_bs["label_en_long_filled"] = merged_bs.label_en_long.replace({"-": None}).fillna(
    label_en_filled,
)
# %%

print(merged_bs["label_jp_long_filled"].isna().sum())

print(merged_bs["sign_lab"].isna().sum())

merged_bs.to_pickle(PROJDIR / "data/3_processed/dataset_2507" / "merged_bs_filled_0714.pkl")
# %%
print("all unique account", merged_bs.label_jp_long_filled.nunique())

# %%
merged_bs_f = merged_bs.query("label_jp_long_filled.notna() and sign_lab.notna()")
# %%

merged_bs_f_g = merged_bs.groupby(["docid"]).agg(
    {"label_jp_long_filled": lambda x: list(x)},
)
# %%
print("all unique account", merged_bs_f.label_jp_long_filled.nunique())
print("all unique account", merged_bs_f.label_en_long_filled.nunique())

# %%
merged_bs_f["label_jp_long_filled_list"] = merged_bs_f["label_jp_long_filled"].apply(
    lambda x: [x],
)

# %%

from gensim.corpora.dictionary import Dictionary

dictionary = Dictionary(merged_bs_f_g.label_jp_long_filled.to_list())
# dictionary.compactify()
dictionary.filter_extremes(no_below=100, no_above=1, keep_n=400000)


# %%
def assign_id(x: list):
    if len(dictionary.doc2bow(x)) > 0:
        return dictionary[dictionary.doc2bow(x)[0][0]]
    return None


merged_bs_f = merged_bs_f.assign(
    label_jp_long_filled_id=merged_bs_f.label_jp_long_filled_list.apply(assign_id),
)
# %% save
#merged_bs_f.to_pickle(PROJDIR / "data/3_processed/dataset_2507" / "merged_bs_f.pkl")
# %%

print(
    "missing rate",
    merged_bs_f.label_jp_long_filled_id.isna().sum() / len(merged_bs_f),
)


# %%
print("filtered majior unique account", merged_bs_f.label_jp_long_filled_id.nunique())
# %% PL

# %%
filename = PROJDIR / "data/3_processed/dataset_2507" / "cy_pl.pkl"
cy_pl = pd.read_pickle(filename)


# %%
cy_pl["sign_lab"] = (cy_pl.debit_flg - cy_pl.credit_flg).replace(
    {1: "debit", -1: "credit", 0: None},
)
cy_pl["sign_lab"].isna().sum()


# %%


sign_lab_dict = (
    cy_pl[["sign_lab", "key"]]
    .query("sign_lab.notna()")
    .groupby("key")
    .agg({"sign_lab": g_argmax})
)
# .drop_duplicates()#.to_dict()
print(len(sign_lab_dict))
cy_pl["sign_lab"] = cy_pl["sign_lab"].fillna(
    cy_pl["key"].map(sign_lab_dict["sign_lab"]),
)
cy_pl["sign_lab"].value_counts()
cy_pl["sign_lab"].isna().sum()


filename = PROJDIR / "data/3_processed/dataset_2507" / "merged_pl.pkl"
merged_pl = pd.read_pickle(filename)
merged_pl["sign_lab"] = (merged_pl.debit_flg - merged_pl.credit_flg).replace(
    {1: "debit", -1: "credit", 0: None},
)
merged_pl["sign_lab"] = merged_pl["sign_lab"].fillna(
    merged_pl["key"].map(sign_lab_dict["sign_lab"]),
)
merged_pl.head()

# %%
labname_dict = (
    cy_pl[["label_jp", "key"]]
    .query("label_jp.notna() and label_jp != '-'")
    .groupby("key")
    .agg({"label_jp": g_argmax})
)
merged_pl["label_jp_filled"] = (
    merged_pl["label_jp"]
    .replace({"-": None})
    .fillna(merged_pl["key"].map(labname_dict["label_jp"]))
)
labname_en_dict = (
    cy_pl[["label_en", "key"]]
    .query("label_en.notna() and label_en != '-'")
    .groupby("key")
    .agg({"label_en": g_argmax})
)
label_en_filled = (
    merged_pl["label_en"]
    .replace({"-": None})
    .fillna(merged_pl["key"].map(labname_en_dict["label_en"]))
)
# %%
print("number of '-' in label_jp_filled", len(merged_pl["label_jp_filled"] == "-"))

print("number of '-' in label_en_filled", len(label_en_filled[label_en_filled == "-"]))

# %%
print(merged_pl["sign_lab"].isna().sum())

print(merged_pl["sign_lab"].value_counts())
# %%
merged_pl["label_jp_long_filled"] = merged_pl.label_jp_long.replace({"-": None}).fillna(
    merged_pl.label_jp_filled,
)  # .value_counts()
merged_pl["label_en_long_filled"] = merged_pl.label_en_long.replace({"-": None}).fillna(
    label_en_filled,
)
# %%
print(merged_pl["label_jp_long_filled"].isna().sum())

print(merged_pl["label_en_long_filled"].isna().sum())

print(merged_pl["sign_lab"].isna().sum())

# %%
mask = merged_pl["sign_lab"].isna()
merged_pl.loc[mask, :].key.value_counts().head(30)

# %%


# %%
merged_pl_f = merged_pl.query("label_jp_long_filled.notna() and sign_lab.notna()")
# %%

merged_pl_f_g = merged_pl_f.groupby(["docid"]).agg(
    {"label_jp_long_filled": lambda x: list(x)},
)

# %%
merged_pl_f["label_jp_long_filled_list"] = merged_pl_f["label_jp_long_filled"].apply(
    lambda x: [x],
)

# %%

from gensim.corpora.dictionary import Dictionary

dictionary = Dictionary(merged_pl_f_g.label_jp_long_filled.to_list())
# dictionary.compactify()
dictionary.filter_extremes(no_below=100, no_above=1, keep_n=400000)
#dictionary.save(str(PROJDIR / "data/3_processed/dataset_2507" / "dictionary_pl.dict"))


# %%
def assign_id(x: list):
    if len(dictionary.doc2bow(x)) > 0:
        return dictionary[dictionary.doc2bow(x)[0][0]]
    return None


merged_pl_f = merged_pl_f.assign(
    label_jp_long_filled_id=merged_pl_f.label_jp_long_filled_list.apply(assign_id),
)
# %% save
# merged_pl_f.to_pickle(PROJDIR / "data/3_processed/dataset_2507" / "merged_pl_f.pkl")
# %%
merged_pl_f


# %%

# %%
pl_balance_df = pl_temp[["balance", "key_cap"]].drop_duplicates().set_index("key_cap")
print("number of missing sign_lab", merged_pl.sign_lab.isna().sum())
merged_pl["sign_lab"] = merged_pl["sign_lab"].fillna(
    merged_pl["key"].map(pl_balance_df["balance"]),
)
# %%
print("number of missing sign_lab", merged_pl.sign_lab.isna().sum())
# %%
merged_pl.to_pickle(PROJDIR / "data/3_processed/dataset_2507" / "merged_pl_filled_0714.pkl")


# %%　##############################################################
# amd
##################################################################
# %%
def get_cy_bs_for_complemetion_amd():
    filename = PROJDIR / "data/3_processed/dataset_2507" / "restatement" / "cy_bs.pkl"
    cy_bs_amd = pd.read_pickle(filename)

    cy_bs_amd["sign_lab"] = (cy_bs_amd.debit_flg - cy_bs_amd.credit_flg).replace(
        {1: "debit", -1: "credit", 0: None},
    )
    cy_bs_amd["sign_lab"].isna().sum()

    sign_lab_dict = (
        cy_bs[["sign_lab", "key"]]
        .query("sign_lab.notna()")
        .groupby("key")
        .agg({"sign_lab": g_argmax})
    )
    # .drop_duplicates()#.to_dict()
    print(len(sign_lab_dict))
    cy_bs_amd["sign_lab"] = cy_bs_amd["sign_lab"].fillna(
        cy_bs_amd["key"].map(sign_lab_dict["sign_lab"]),
    )
    cy_bs_amd["sign_lab"].value_counts()
    cy_bs_amd["sign_lab"].isna().sum()
    return cy_bs_amd, sign_lab_dict


cy_bs_amd, sign_lab_dict = get_cy_bs_for_complemetion_amd()

# %%
filename = PROJDIR / "data/3_processed/dataset_2507" / "restatement" / "merged_bs.pkl"
merged_bs_amd = pd.read_pickle(filename)
merged_bs_amd["sign_lab"] = (
    merged_bs_amd.debit_flg - merged_bs_amd.credit_flg
).replace(
    {1: "debit", -1: "credit", 0: None},
)
merged_bs_amd["sign_lab"] = merged_bs_amd["sign_lab"].fillna(
    merged_bs_amd["key"].map(sign_lab_dict["sign_lab"]),
)
merged_bs_amd.head()

# fillna_1 (Excelで補う)
bs_balance_df = bs_temp[["balance", "key_cap"]].drop_duplicates().set_index("key_cap")
print("number of missing sign_lab", merged_bs_amd.sign_lab.isna().sum())
merged_bs_amd["sign_lab"] = merged_bs_amd["sign_lab"].fillna(
    merged_bs_amd["key"].map(bs_balance_df["balance"]),
)

# %% fillna_2 (他の企業のlabelで欠損を補完)
labname_dict = (
    cy_bs[["label_jp", "key"]]
    .query("label_jp.notna()")
    .groupby("key")
    .agg({"label_jp": g_argmax})
)
merged_bs_amd["label_jp_filled"] = merged_bs_amd["label_jp"].fillna(
    merged_bs_amd["key"].map(labname_dict["label_jp"]),
)
labname_en_dict = (
    cy_bs[["label_en", "key"]]
    .query("label_en.notna() and label_en != '-'")
    .groupby("key")
    .agg({"label_en": g_argmax})
)
label_en_filled = (
    merged_bs_amd["label_en"]
    .replace({"-": None})
    .fillna(merged_bs_amd["key"].map(labname_en_dict["label_en"]))
)
# %%
print(merged_bs_amd["sign_lab"].value_counts())

# %%
merged_bs_amd["label_jp_long_filled"] = merged_bs_amd.label_jp_long.replace(
    {"-": None},
).fillna(
    merged_bs_amd.label_jp_filled,
)  # .value_counts()
merged_bs_amd["label_en_long_filled"] = merged_bs_amd.label_en_long.replace(
    {"-": None},
).fillna(
    label_en_filled,
)
# %%
merged_bs_amd.query("docid == 'S100X5AB'")

# %%
print(merged_bs_amd["label_jp_long_filled"].isna().sum())

print(merged_bs_amd["sign_lab"].isna().sum())

merged_bs_amd.to_pickle(
    PROJDIR / "data/3_processed/dataset_2507" / "restatement" / "merged_bs_filled_0714.pkl",
)

# %%
# %%
filename = PROJDIR / "data/3_processed/dataset_2507" / "restatement" / "cy_pl.pkl"
cy_pl_amd = pd.read_pickle(filename)

# %%
cy_pl_amd["sign_lab"] = (cy_pl_amd.debit_flg - cy_pl_amd.credit_flg).replace(
    {1: "debit", -1: "credit", 0: None},
)
cy_pl_amd["sign_lab"].isna().sum()


# %%
def g_argmax(x):
    return x.value_counts().idxmax()


sign_lab_dict = (
    cy_pl[["sign_lab", "key"]]
    .query("sign_lab.notna()")
    .groupby("key")
    .agg({"sign_lab": g_argmax})
)
# .drop_duplicates()#.to_dict()
print(len(sign_lab_dict))
cy_pl_amd["sign_lab"] = cy_pl_amd["sign_lab"].fillna(
    cy_pl_amd["key"].map(sign_lab_dict["sign_lab"]),
)
cy_pl_amd["sign_lab"].value_counts()
cy_pl_amd["sign_lab"].isna().sum()


filename = PROJDIR / "data/3_processed/dataset_2507" / "restatement" / "merged_pl.pkl"
merged_pl_amd = pd.read_pickle(filename)
merged_pl_amd["sign_lab"] = (
    merged_pl_amd.debit_flg - merged_pl_amd.credit_flg
).replace(
    {1: "debit", -1: "credit", 0: None},
)
merged_pl_amd["sign_lab"] = merged_pl_amd["sign_lab"].fillna(
    merged_pl_amd["key"].map(sign_lab_dict["sign_lab"]),
)
merged_pl_amd.head()

# %%
labname_dict = (
    cy_pl[["label_jp", "key"]]
    .query("label_jp.notna() and label_jp != '-'")
    .groupby("key")
    .agg({"label_jp": g_argmax})
)
merged_pl_amd["label_jp_filled"] = (
    merged_pl_amd["label_jp"]
    .replace({"-": None})
    .fillna(merged_pl_amd["key"].map(labname_dict["label_jp"]))
)
labname_en_dict = (
    cy_pl[["label_en", "key"]]
    .query("label_en.notna() and label_en != '-'")
    .groupby("key")
    .agg({"label_en": g_argmax})
)
label_en_filled = (
    merged_pl_amd["label_en"]
    .replace({"-": None})
    .fillna(merged_pl_amd["key"].map(labname_en_dict["label_en"]))
)
# %%
print("number of '-' in label_jp_filled", len(merged_pl_amd["label_jp_filled"] == "-"))
# %%
print(merged_pl_amd["sign_lab"].isna().sum())

print(merged_pl_amd["sign_lab"].value_counts())
# %%
merged_pl_amd["label_jp_long_filled"] = merged_pl_amd.label_jp_long.replace(
    {"-": None},
).fillna(
    merged_pl_amd.label_jp_filled,
)  # .value_counts()
merged_pl_amd["label_en_long_filled"] = merged_pl_amd.label_en_long.replace(
    {"-": None},
).fillna(
    label_en_filled,
)
# %%
print(merged_pl_amd["label_jp_long_filled"].isna().sum())
print(merged_pl_amd["sign_lab"].isna().sum())

# %%
mask = merged_pl_amd["sign_lab"].isna()
merged_pl_amd.loc[mask, :].key.value_counts().head(30)

# %%


# %%
merged_pl_f_amd = merged_pl_amd.query(
    "label_jp_long_filled.notna() and sign_lab.notna()",
)
# %%

merged_pl_f_g_amd = merged_pl_f_amd.groupby(["docid"]).agg(
    {"label_jp_long_filled": lambda x: list(x)},
)

# %%
merged_pl_f_amd["label_jp_long_filled_list"] = merged_pl_f_amd[
    "label_jp_long_filled"
].apply(
    lambda x: [x],
)

# %%

from gensim.corpora.dictionary import Dictionary

dictionary = Dictionary(merged_pl_f_g_amd.label_jp_long_filled.to_list())
# dictionary.compactify()
dictionary.filter_extremes(no_below=100, no_above=1, keep_n=400000)
#dictionary.save(
#    str(
#        PROJDIR
#        / "data/3_processed/dataset_2507"
#        / "restatement"
#        / "dictionary_pl.dict",
#    ),
#)


# %%
def assign_id(x: list):
    if len(dictionary.doc2bow(x)) > 0:
        return dictionary[dictionary.doc2bow(x)[0][0]]
    return None


merged_pl_f_amd = merged_pl_f_amd.assign(
    label_jp_long_filled_id=merged_pl_f_amd.label_jp_long_filled_list.apply(assign_id),
)
# %% save
#merged_pl_f_amd.to_pickle(
#    PROJDIR / "data/3_processed/dataset_2507" / "restatement" / "merged_pl_f_amd.pkl",
#)
# %%


# %%
bs_pl_table_loader = bs_pl_table()
bs_temp, pl_temp, oci_temp, ss_temp, cf_temp = bs_pl_table_loader.get_bs_pl_all()

# %%
pl_balance_df = pl_temp[["balance", "key_cap"]].drop_duplicates().set_index("key_cap")
print("number of missing sign_lab", merged_pl.sign_lab.isna().sum())
merged_pl_amd["sign_lab"] = merged_pl_amd["sign_lab"].fillna(
    merged_pl_amd["key"].map(pl_balance_df["balance"]),
)
# %%
print("number of missing sign_lab", merged_pl_amd.sign_lab.isna().sum())
# %%
merged_pl_amd.to_pickle(
    PROJDIR / "data/3_processed/dataset_2507" / "restatement" / "merged_pl_filled_0714.pkl",
)

# %%
