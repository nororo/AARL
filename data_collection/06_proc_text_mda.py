# %%
import warnings

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# import torch
import json
import time
from pathlib import Path

# %%
PROJPATH = r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR = Path(PROJPATH)

# .tag.value_counts()
# %% 20250128 MD and A 抽出 batch作成
import pickle
import sys

sys.path.append(r"/Users/noro/Documents/Projects/XBRL_common_space_projection")
sys.path.append(
    r"/Users/noro/Documents/Projects/XBRL_common_space_projection/src/edinet_xbrl_prep",
)
from langchain.text_splitter import CharacterTextSplitter
from src.data import data_utils
from src.data.libs.compose_prompt import *
from src.data.libs.model_api import *
from src.data.libs.utils import *
from src.data.libs.xbrl_prep_patch import *

from edinet_xbrl_prep.link_base_file_analyzer import *


def make_batch(dict_df, out_filename, prompt_gen_func, model_name="gpt_4o_mini"):
    batch_inf_file_generator_obj = batch_inf_file_generator(
        model_name=model_name,
    )

    for index_num in tqdm(dict_df.index):
        # description_text=dict_df.loc[index_num,'description']
        dict_df.loc[index_num, :]
        sys_prompt, usr_prompt = prompt_gen_func(dict_df.loc[index_num, :])
        itr_index_str = str(index_num)
        batch_inf_file_generator_obj.insert_inf_list_prompt(
            sys_prompt,
            usr_prompt,
            itr_index_str,
            max_tokens=1024,
            model_name=model_name,
        )

    # out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_"+model_name+".jsonl")
    batch_inf_file_generator_obj.export_list(out_filename)
    batch_inf_file_generator_obj.print_sample()


def preproc_nlp_xbrltext(text: str) -> str:
    # unicode
    replaced_text = unicodedata.normalize("NFKC", text)
    # drop number
    # replaced_text = drop_number(replaced_text)
    # drop signature 1
    # replaced_text = re.sub(re.compile("[!-/:-@[-`{-~]"), '', replaced_text)
    # drop signature 2
    # replaced_text = re.sub(r'\(', '', replaced_text)
    # replaced_text=replaced_text.replace('。','\n')
    # drop signature 3
    # table = str.maketrans("", "", string.punctuation  + "◆■※")
    # replaced_text = replaced_text.translate(table)
    # drop return (recursive)
    replaced_text = data_utils.RtnDroper(replaced_text)

    return replaced_text


def multi_stage_chunk(text: str, max_text_length=20000) -> list:
    """1. by section ('\n\n')
    2. by sentence ('\n')
    3. by double space ('  ')
    4. by single space (' ')
    """
    text_splitter1 = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=max_text_length,
        chunk_overlap=0,
    )
    text_splitter2 = CharacterTextSplitter(
        separator="\n",
        chunk_size=max_text_length,
        chunk_overlap=0,
    )
    text_splitter3 = CharacterTextSplitter(
        separator="  ",
        chunk_size=max_text_length,
        chunk_overlap=0,
    )
    text_splitter4 = CharacterTextSplitter(
        separator=" ",
        chunk_size=max_text_length,
        chunk_overlap=0,
    )

    text_list = text_splitter1.split_text(text)
    text_list_out = []
    for itr in range(len(text_list)):
        text_list_out = text_list_out + text_splitter2.split_text(text_list[itr])
    text_list = text_list_out
    text_list_out = []
    for itr in range(len(text_list)):
        text_list_out = text_list_out + text_splitter3.split_text(text_list[itr])
    text_list = text_list_out
    text_list_out = []
    for itr in range(len(text_list)):
        text_list_out = text_list_out + text_splitter4.split_text(text_list[itr])
    return text_list_out


def make_prompt_qag_prep(prompt_dict: dict, sample_text: str):
    instruction = prompt_dict["extsummary_instruction"]

    constraints = prompt_dict["extsummary_constraints"]
    preface_constraint = "#### 注意事項"
    bullet_char = " * "
    constraint = (
        preface_constraint + "\n" + bullet_char + ("\n" + bullet_char).join(constraints)
    )
    example = prompt_dict["extsummary_example"]
    output_format = prompt_dict["extsummary_output_formats"]
    system_prompt_comp = (
        instruction + "\n\n" + constraint + "\n\n" + output_format + "\n\n" + example
    )
    user_prompt_comp = "#### 文章" + "\n" + sample_text + "\n\n" + "#### 回答" + "\n"
    return system_prompt_comp, user_prompt_comp


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


def get_data(
    account_link_tracer_obj,
    xbrl_obj,
    include_tree_top_keyword,
    role_text,
    include_keyword_list,
    exclude_keyword,
    keep_taxonomi_list,
    docid,
    edinet_code,
    period_end,
    period_start,
    tag,
):
    log_dict = {}
    all_keys1 = []
    dict_concat = []
    try:
        for key in account_link_tracer_obj.search_keys(include_tree_top_keyword):
            all_keys1 = (
                all_keys1
                + [key]
                + account_link_tracer_obj.get_child_keys_recursive(key, role_text)
            )
        key_list_pre = list(set(all_keys1))

        if len(include_keyword_list) > 0:
            all_keys2 = []
            for keyword in include_keyword_list:
                all_keys2 = all_keys2 + account_link_tracer_obj.search_keys(keyword)
            key_list_pre = list(set(key_list_pre) & set(all_keys2))

        if len(exclude_keyword) > 0:
            all_exclude_keys = account_link_tracer_obj.search_keys(exclude_keyword)
            key_list_pre = list(set(key_list_pre) - set(all_exclude_keys))

        key_list_pre = [
            key for key in key_list_pre if ("TextBlock" in key) | ("Heading" in key)
        ]

        if len(key_list_pre) > 0:
            order_df = account_link_tracer_obj.get_child_order_recursive_list(
                key_list=key_list_pre,
                role=role_text,
            )
            key_list = list(filter(lambda x: x in set(xbrl_obj.key), key_list_pre))
        else:
            key_list = keep_taxonomi_list
            order_df = pd.DataFrame(columns=["child_key", "role", "order"])

        xbrl_obj_com = (
            xbrl_obj.query("key in @key_list").sort_values("order").reset_index()
        )
        # xbrl_obj_com=pd.merge(xbrl_obj_com,order_df,left_on='key',right_on='child_key',how='left').sort_values('order')

        if len(xbrl_obj_com) > 0:
            text_all = "\n\n".join(
                xbrl_obj_com.data_str.apply(data_utils.htmldrop)
                .apply(preproc_nlp_xbrltext)
                .to_list(),
            )
            text_list = multi_stage_chunk(text_all, max_text_length=5000)
            itr_num = 0
            for text in text_list:
                dict_t = {
                    "docID": docid,
                    "nb": itr_num,
                    "edinet_code": edinet_code,
                    "period_end": period_end,
                    "period_start": period_start,
                    "tag": tag,
                    "text": text,
                }
                itr_num = itr_num + 1
                dict_concat.append(dict_t)
            log_dict["get_data_status_" + tag] = "Success"
            log_dict["get_data_error_message_" + tag] = None
        else:
            log_dict["get_data_status_" + tag] = "Failure"
            log_dict["get_data_error_message_" + tag] = "No data"
    except Exception as e:
        log_dict["get_data_status_" + tag] = "Failure"
        log_dict["get_data_error_message_" + tag] = repr(e)

    return dict_concat, log_dict


def preproc_fs(zip_file_str: str, itr_docID, response_tbl, role_keyward_list: list):
    temp_path_str = str(
        PROJDIR
        / "data"
        / "2_intermediate"
        / (f"data_pool_{response_tbl.loc[itr_docID, 'dataset']}")
        / itr_docID,
    )
    filename = temp_path_str + "/fs_tbl.pkl"
    if Path(filename).is_file() == False:
        print(f"file not found: {filename}")
    fs_tbl = FsDataDf(pd.read_pickle(filename))
    log_filename = temp_path_str + "/fs_tbl_log.json"
    log_dict = json.load(open(log_filename))
    # print(json.dumps(log_dict, indent=4))
    conf = {
        "tag": "anly",
        "include_tree_top_keyword": "OverviewOfBusinessHeading",
        "include_keyword_list": ["ManagementAnalysis"],  # CriticalContract
        "exclude_keyword": [],
        "keep_taxonomi_list": [
            "jpcrp_cor:ManagementAnalysisOfFinancialPositionOperatingResultsAndCashFlowsHeading",
            "jpcrp_cor:ManagementAnalysisOfFinancialPositionOperatingResultsAndCashFlowsTextBlock",
            #'jpcrp_cor:AnalysisAndResponsesToSignificantEventsRelatedToGoingConcernRisksEtcTextBlock'
            #'jpcrp_cor:CriticalContractsForOperationHeading',
            #'jpcrp_cor:CriticalContractsForOperationTextBlock'
        ],
    }
    # get linkbase file
    linkbasefile_obj = linkbasefile_cor(
        zip_file_str=zip_file_str,
        temp_path_str=temp_path_str,
    )
    linkbasefile_obj.read_linkbase_file()
    # make account label
    # linkbasefile_obj.make_account_label(
    #    account_list_common_obj=account_list_common_obj,
    #    role_list=role_keyward_list
    #    )
    account_link_tracer_obj = account_link_tracer_loc(linkbasefile_obj.parent_child_df)
    # account_link_tracer_obj=linkbasefile_obj.account_link_tracer_obj
    role_text = "http://disclosure.edinet-fsa.go.jp/role/jpcrp/rol_CabinetOfficeOrdinanceOnDisclosureOfCorporateInformationEtcFormNo3AnnualSecuritiesReport"
    edinet_code = response_tbl.loc[itr_docID, "response_edinetCode"]
    period_start = response_tbl.loc[itr_docID, "response_periodStart"]
    period_end = response_tbl.loc[itr_docID, "response_periodEnd"]

    dict_add, log_dict = get_data(
        account_link_tracer_obj=account_link_tracer_obj,
        xbrl_obj=fs_tbl,
        include_tree_top_keyword=conf["include_tree_top_keyword"],
        role_text=role_text,
        include_keyword_list=conf["include_keyword_list"],
        exclude_keyword=conf["exclude_keyword"],
        keep_taxonomi_list=conf["keep_taxonomi_list"],
        docid=itr_docID,
        edinet_code=edinet_code,
        period_end=period_end,
        period_start=period_start,
        tag=conf["tag"],
    )
    # result
    preproc_log = {}

    # response_tbl
    preproc_log["docid"] = itr_docID
    preproc_log["edinet_code"] = response_tbl.loc[itr_docID, "response_edinetCode"]
    preproc_log["sec_code"] = response_tbl.loc[itr_docID, "response_secCode"]
    preproc_log["period_start"] = response_tbl.loc[itr_docID, "response_periodStart"]
    preproc_log["period_end"] = response_tbl.loc[itr_docID, "response_periodEnd"]
    preproc_log["EDINET_taxonomy_year"] = response_tbl.loc[itr_docID, "year"]

    return dict_add  # ,log_dict#, kpi_val, cy_pl, cy_bs, py_pl, py_bs


import joblib


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


def get_zipdir2(docid: str):
    try:
        data_dir_raw = PROJDIR / "data" / "1_raw"
        zip_file = list(data_dir_raw.glob("data_pool_*/" + docid + ".zip"))[0]
    except Exception:
        zip_file = "not found"
    return zip_file


# %%

example_text_risk3 = """
#### 例

##### 文章

4【経営者による財政状態、経営成績及びキャッシュ・フローの状況の分析】
当連結会計年度における当社グループ(当社及び連結子会社)の財政状態、経営成績及びキャッシュ・フロー(以下、「経営成績等」という。)の状況の概要並びに経営者の視点による当社グループの経営成績等の状況に関する認識及び分析・検討内容は次のとおりであります。なお、文中の将来に関する事項は、当連結会計年度末現在において判断したものであります。
また、当社グループは、電子部品事業の単一セグメントであるため、セグメント情報に関連付けた記載を行っておりません。
(1)経営成績
当連結会計年度(2023年4月1日から2024年3月31日まで)における当社グループを取り巻く経営環境は、世界景気は緩やかな持ち直しの動きが見られたものの、一部地域において弱さが見られるなど不透明な状況が続きました。先行きについては、国際情勢、金融資本市場の変動などを注視する必要がありますが、緩やかな回復が続くことが期待されます。
当社グループは、中期経営計画2025に掲げた目標の実現に向けて自動車、情報インフラ・産業機器を中心とした注力すべき市場の売上比率を50%とすることを目指しています。さらに、ハイエンド商品、高信頼性商品を中心とした高付加価値な電子部品を創出し、主力事業の積層セラミックコンデンサのさらなる成長に加え、インダクタと通信デバイスを強化してコア事業として確立していきます。また、需要拡大に対応するための継続的な能力増強に加え、環境対策やIT整備に向けた積極的な取り組みを実施し、5年間で3,000億円規模の設備投資を計画しています。
当連結会計年度の連結売上高は3,226億47百万円(前年同期比1.0%増)、営業利益は90億79百万円(前年同期比71.6%減)、経常利益は137億57百万円(前年同期比60.5%減)、親会社株主に帰属する当期純利益は83億17百万円(前年同期比64.2%減)となりました。情報機器、情報インフラ・産業機器などを中心とした生産台数の減少や在庫調整などにより、各段階利益が減少しました。
当連結会計年度における期中平均の為替レートは1米ドル143.32円と前年同期の平均為替レートである1米ドル134.20円と比べ9.12円の円安となりました。
製品別の売上高は次のとおりであります。
[コンデンサ]
積層セラミックコンデンサなどが含まれます。
当連結会計年度は、通信機器、自動車向けの売上が前年同期比で増加しましたが、民生機器、情報機器、情報インフラ・産業機器向けの売上が前年同期比で減少したことにより、売上高は2,058億29百万円(前年同期比1.1%減)となりました。
[インダクタ]
巻線インダクタ、積層インダクタなどの各種インダクタ商品が含まれます。
当連結会計年度は、民生機器、情報インフラ・産業機器向けの売上が前年同期比で減少しましたが、情報機器、通信機器、自動車向けの売上が前年同期比で増加したことにより、売上高は555億66百万円(前年同期比5.1%増)となりました。
[複合デバイス]
通信用デバイス(FBAR/SAW)、回路モジュールなどが含まれます。
当連結会計年度は、回路モジュールの売上が前年同期比で減少しましたが、通信用デバイス(FBAR/SAW)の売上が前年同期比で増加したことにより、売上高は349億34百万円(前年同期比7.2%増)となりました。
[その他]
アルミニウム電解コンデンサなどが含まれます。
当連結会計年度は、アルミニウム電解コンデンサの売上が前年同期比で増加したことにより、売上高は263億17百万円(前年同期比1.4%増)となりました。
生産、受注及び販売の実績
1生産実績
当連結会計年度における生産実績を製品別に示すと、次のとおりであります。
 製品別
生産高(百万円)
前年同期比(%)
コンデンサ
207,498
0.8
(注) 金額は、期中の平均販売単価を用いております。
2受注実績
当連結会計年度における受注実績を製品別に示すと、次のとおりであります。
△9.7
3販売実績
当連結会計年度における販売実績を製品別に示すと、次のとおりであります。
 製品別
販売高(百万円)
前年同期比(%)
コンデンサ
205,829
△1.1
(注) 主要な販売先は、当該販売実績の総販売実績に対する割合が100分の10未満であるため記載を省略しております。
(2)財政状態
1 資産
当連結会計年度末における総資産の残高は5,796億86百万円となり、前連結会計年度末に比べ762億23百万円増加しました。流動資産は228億23百万円増加しており、主な要因は、現金及び預金の増加155億85百万円、受取手形及び売掛金の増加108億26百万円、仕掛品の減少18億91百万円、商品及び製品の減少14億8百万円であります。また、固定資産は534億円増加しており、主な要因は、有形固定資産の増加529億66百万円であります。
...

##### 回答
{"経営環境や会社の行動":"世界景気は緩やかな持ち直しの動きが見られたものの、一部地域において弱さが見られる不透明な状況が続いた","会計数値への影響":"記載なし"}
{"経営環境や会社の行動":"中期経営計画2025に基づき、自動車、情報インフラ・産業機器を中心とした注力市場の売上比率50%を目指す","会計数値への影響":"記載なし"}
{"経営環境や会社の行動":"情報機器、情報インフラ・産業機器などを中心とした生産台数の減少や在庫調整","会計数値への影響":"営業利益が前年同期比71.6%減の90億79百万円、経常利益が前年同期比60.5%減の137億57百万円、親会社株主に帰属する当期純利益が前年同期比64.2%減の83億17百万円となった"}
{"経営環境や会社の行動":"為替レートが1米ドル134.20円から143.32円へ円安となった","会計数値への影響":"記載なし"}
{"経営環境や会社の行動":"通信機器、自動車向けの売上増加、民生機器、情報機器、情報インフラ・産業機器向けの売上減少[コンデンサ部門]","会計数値への影響":"売上高が前年同期比1.1%減の2,058億29百万円となった"}
{"経営環境や会社の行動":"情報機器、通信機器、自動車向けの売上増加、民生機器、情報インフラ・産業機器向けの売上減少[インダクタ部門]","会計数値への影響":"売上高が前年同期比5.1%増の555億66百万円となった"}
{"経営環境や会社の行動":"通信用デバイス(FBAR/SAW)の売上増加、回路モジュールの売上減少[複合デバイス部門]","会計数値への影響":"売上高が前年同期比7.2%増の349億34百万円となった"}
{"経営環境や会社の行動":"アルミニウム電解コンデンサの売上増加[その他部門]","会計数値への影響":"売上高が前年同期比1.4%増の263億17百万円となった"}


以上の指示に従って、提供された文章から、経営環境や会社の行動を抽出して整理してください。

"""

memo = """
2 負債
当連結会計年度末における負債の残高は2,495億87百万円となり、前連結会計年度末に比べ646億3百万円増加しました。主な要因は、転換社債型新株予約権付社債の増加511億70百万円、長期借入金の増加334億41百万円、支払手形及び買掛金の増加67億32百万円、短期借入金の減少260億円、1年内返済予定の長期借入金の減少87億2百万円であります。
3 純資産
当連結会計年度末における純資産の残高は3,300億98百万円となり、前連結会計年度末に比べ116億20百万円増加しました。主な要因は、親会社株主に帰属する当期純利益83億17百万円と剰余金の配当112億15百万円による、利益剰余金の減少28億97百万円、及び円安等の為替影響による為替換算調整勘定の増加147億74百万円であります。
(3)キャッシュ・フローの状況の分析・検討内容並びに資本の財源及び資金の流動性に係る情報
当連結会計年度の営業活動によるキャッシュ・フローは511億4百万円の収入(前年同期比29.5%増)となりました。主な要因は、税金等調整前当期純利益130億73百万円、減価償却費393億91百万円、棚卸資産の減少額71億46百万円、売上債権の増加額48億40百万円であります。
投資活動によるキャッシュ・フローは827億93百万円の支出(前年同期比37.0%増)となりました。主な要因は、固定資産の取得による支出799億7百万円であります。
財務活動によるキャッシュ・フローは376億47百万円の収入(前年同期比159.9%増)となりました。主な要因は、転換社債型新株予約権付社債の発行による収入511億33百万円、長期借入れによる収入427億8百万円、短期借入金の減少額260億円、長期借入金の返済による支出179億69百万円、配当金の支払額111億98百万円であります。
以上の結果、当連結会計年度末における現金及び現金同等物は、前連結会計年度末に対して108億15百万円増加し、949億40百万円となりました。
当連結会計年度末の外部からの資金調達は、短期借入金42億円、1年内返済予定の長期借入金92億55百万円、転換社債型新株予約権付社債511億70百万円、長期借入金842億19百万円からなっております。借入金は原則として日本において固定金利で調達しております。さらに、財務の安定性のため期間3年、300億円のコミットメントライン借入枠を設定しておりますが、2024年3月末現在未使用であります。
当社グループは、健全な財務状態と営業活動によりキャッシュ・フローを生み出す能力を有しており、当社グループの成長を維持するために将来必要な運転資金及び設備投資資金を調達することが可能と考えております。
(4)経営上の目標の達成・進捗状況
当社グループは、2021年度を初年度とする「中期経営計画2025」を策定しており、目標とする経営指標は「第2 事業の状況 1 経営方針、経営環境及び対処すべき課題等 (2)中長期的な会社の経営戦略と目標とする経営指標」に記載のとおりであります。当連結会計年度における連結売上高は3,226億47百万円、営業利益率は2.8%、ROE(自己資本利益率)は2.6%、ROIC(投下資本利益率)は1.9%となりました。連結売上高の目標4,800億円は、最終年度である2025年度までの達成を目指し、事業成長や経営の効率化に取り組んでまいります。
(5)重要な会計上の見積り及び当該見積りに用いた仮定
連結財務諸表の作成に当たって用いた会計上の見積り及び当該見積りに用いた仮定のうち、重要なものについては、「第5 経理の状況 1 連結財務諸表等 (1)連結財務諸表 注記事項(重要な会計上の見積り)」に記載のとおりであります。

"""


prompt_ext_risk2 = {
    "extsummary_instruction": """提供される文章はある経営者による財政状態、経営成績及びキャッシュ・フローの状況の分析の抜粋です。経営環境や会社の行動と、それによる会計数値への影響を日本語で抽出し整理してください。""",
    "extsummary_example": example_text_risk3,
    #### 注意事項
    "extsummary_constraints": [
        "経営環境や会社の行動（原因）と会計数値への影響（結果）に分けてください（会計数値への影響の記載があれば）",
        "該当する記載がない場合は「記載なし」としてください。",
        "固有名詞は伏せてください。",
    ],
    "extsummary_output_formats": """#### 回答形式\n\nフォーマットは個別のjson形式で回答してください。\n\n{"経営環境や会社の行動":"(経営環境や会社の行動1)","会計数値への影響":"(会計数値への影響1)"}\n{"経営環境や会社の行動":"(経営環境や会社の行動2)","会計数値への影響":"(会計数値への影響2)"}""",
    #### 文章
    # ${}
}

prompt_ext = {"anly": prompt_ext_risk2}

# %%
TESTDIR = Path(PROJPATH) / "tests/20250115"
model_name = "llama_3.1_70b"
tag_name = "anly"
# filename = TESTDIR / "account_list_common_obj_dict.pkl"
# with open(filename, 'rb') as f:
#    account_list_common_obj_dict = pickle.load(f)

fs_dict = {
    "BS": ["_BalanceSheet", "_ConsolidatedBalanceSheet"],
    "PL": ["_StatementOfIncome", "_ConsolidatedStatementOfIncome"],
    "CF": ["_StatementOfCashFlows", "_ConsolidatedStatementOfCashFlows"],
    "SS": ["_StatementOfChangesInEquity", "_ConsolidatedStatementOfChangesInEquity"],
    "notes": ["_Notes", "_ConsolidatedNotes"],
    "report": ["_CabinetOfficeOrdinanceOnDisclosure"],
}

filename = PROJPATH + "data/3_processed/dataset_2407/response_tbl_with_year.pkl"
response_tbl = pd.read_pickle(filename)

filename = PROJPATH + "data/3_processed/dataset_2507/response_tbl_with_year_add.pkl"
response_tbl_add = pd.read_pickle(filename)
response_tbl_add = response_tbl_add.query("year != '-'")
assert len(set(response_tbl_add.index) & set(response_tbl.index)) == 0

response_tbl = pd.concat([response_tbl, response_tbl_add])


n_parallel = 16
with tqdm_joblib(
    tqdm(desc="My calculation", total=len(response_tbl.index)),
) as progress_bar:
    results = joblib.Parallel(n_jobs=n_parallel, verbose=0)(
        [
            joblib.delayed(preproc_fs)(
                zip_file_str=get_zipdir2(itr_docID),
                itr_docID=itr_docID,
                response_tbl=response_tbl,
                role_keyward_list=fs_dict["report"],
            )
            for itr_docID in response_tbl.index
        ],
    )
# %%
# save
filename = PROJPATH + "data/3_processed/dataset_2507/extracted_text_mda.pkl"
with open(filename, "wb") as f:
    pickle.dump(results, f)

# %%
rst_list = []
batch_inf_file_generator_obj = batch_inf_file_generator(
    model_name=model_name,
)


for result_list in results:
    for itr_num, text_dict in enumerate(result_list):
        text = text_dict["text"]

        # extract SoI
        pl_sect = [
            itr for itr, line in enumerate(text.split("\n")) if ("経営成績" in line)
        ]
        if len(pl_sect) == 0:
            start_idx = 0
        else:
            start_idx = max(pl_sect)
        bs_sect = [
            itr
            for itr, line in enumerate(text.split("\n"))
            if ("財政状態" in line) & (itr > start_idx)
        ]
        if len(bs_sect) > 0:
            start_other = min(bs_sect)
            end_idx = max(start_idx + 10, start_other)
            proc_text = "\n".join(
                [line for line in text.split("\n")][start_idx : end_idx + 1],
            )
        else:
            proc_text = "\n".join([line for line in text.split("\n")][start_idx:])
        # proc_text = "\n".join([line for line in text.split("\n") if (('成績' in line)&('経営' in line))|(len(drop_number(line))>10) ])
        itr_docID = text_dict["docID"]
        sys_prompt, usr_prompt = make_prompt_qag_prep(prompt_ext[tag_name], proc_text)
        itr_index_str = str(itr_docID) + "_" + str(tag_name) + "_" + str(itr_num)
        batch_inf_file_generator_obj.insert_inf_list_prompt(
            sys_prompt,
            usr_prompt,
            itr_index_str,
            max_tokens=2048,
            model_name=model_name,
        )

out_filename = (
    PROJDIR
    / "data/3_processed/dataset_2507/llm_proc"
    / ("batch_extract_text_" + model_name + "_anly_act_cor2.jsonl")
)
batch_inf_file_generator_obj.export_list(out_filename)
batch_inf_file_generator_obj.print_sample()
# %%

# if __name__ == "__main__":
#    print("start")
#    main()
#    print("end")

# %%
# %%
# %% llama 405B
model_name = "llama_405b"
rst_list = []
batch_inf_file_generator_obj = batch_inf_file_generator(
    model_name=model_name,
)


for result_list in results:
    if len(result_list) == 0:
        continue
    for itr_num, text_dict in enumerate(result_list):
        text = text_dict["text"]

        itr_docID = text_dict["docID"]
        sys_prompt, usr_prompt = make_prompt_qag_prep(prompt_ext[tag_name], text)
        itr_index_str = str(itr_docID) + "_" + str(tag_name) + "_" + str(itr_num)
        batch_inf_file_generator_obj.insert_inf_list_prompt(
            sys_prompt,
            usr_prompt,
            itr_index_str,
            max_tokens=1024,
            model_name=model_name,
        )

out_filename = (
    PROJDIR
    / "data/3_processed/dataset_2507/llm_proc"
    / ("batch_extract_text_" + model_name + "_anly_act.jsonl")
)
batch_inf_file_generator_obj.export_list(out_filename)
batch_inf_file_generator_obj.print_sample()

# %%
print(len(batch_inf_file_generator_obj.inf_list))
# %%
