# %%
import sys

sys.path.append(r"/Users/noro/Documents/Projects/XBRL_common_space_projection")
sys.path.append(
    r"/Users/noro/Documents/Projects/XBRL_common_space_projection/src/edinet_xbrl_prep",
)

import warnings

import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from src.data import data_utils
from src.data.libs.compose_prompt import *
from src.data.libs.model_api import *
from src.data.libs.utils import *
from src.data.libs.xbrl_prep_patch import *
from tqdm import tqdm

from edinet_xbrl_prep.link_base_file_analyzer import *

warnings.filterwarnings("ignore")

# import torch
import json
import time
from os.path import dirname, join
from pathlib import Path

from dotenv import load_dotenv

# %%
PROJPATH = r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR = Path(PROJPATH)

# %%
# from openai import OpenAI
load_dotenv(verbose=True)
dotenv_path = join(Path(dirname(__file__)).parents[1] / "env" / "k", ".env")
load_dotenv(dotenv_path)

# %%

######################################################
#
#            Prompt
#
######################################################
from src.data.libs.compose_prompt import *
from src.data.libs.utils import *


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


def preproc_fs(
    zip_file_str: str,
    itr_docID,
    response_tbl,
    role_keyward_list: list,
    conf: dict,
):
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
    # conf = {
    #    "tag": "anly",
    #    "include_tree_top_keyword": "OverviewOfBusinessHeading",
    #    "include_keyword_list": ["ManagementAnalysis"],  # CriticalContract
    #    "exclude_keyword": [],
    #    "keep_taxonomi_list": [
    #        "jpcrp_cor:ManagementAnalysisOfFinancialPositionOperatingResultsAndCashFlowsHeading",
    #        "jpcrp_cor:ManagementAnalysisOfFinancialPositionOperatingResultsAndCashFlowsTextBlock",
    #        #'jpcrp_cor:AnalysisAndResponsesToSignificantEventsRelatedToGoingConcernRisksEtcTextBlock'
    #        #'jpcrp_cor:CriticalContractsForOperationHeading',
    #        #'jpcrp_cor:CriticalContractsForOperationTextBlock'
    #    ],
    # }
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
# NEXT: Long textの処理
#  -> chunk 項目指定で抽出


def preproc_nlp_xbrltext(text: str) -> str:
    # unicode
    replaced_text = unicodedata.normalize("NFKC", text)
    # drop number
    # replaced_text = drop_number(replaced_text)
    # drop signature 1
    replaced_text = re.sub(re.compile("[!-/:-@[-`{-~]"), "", replaced_text)
    # drop signature 2
    replaced_text = re.sub(r"\(", "", replaced_text)
    replaced_text = replaced_text.replace("。", "\n")
    # drop signature 3
    table = str.maketrans("", "", string.punctuation + "◆■※")
    replaced_text = replaced_text.translate(table)
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


# %%
# Business
example_text = """
#### 例
##### 文章
3 【事業の内容】当社の主たる事業は物流業であります
その事業は貨物運送事業、倉庫事業、その他事業に区分されますが、それぞれの事業内容は次のとおりであります
イ 貨物運送事業貨物自動車運送事業法に基づく、一般貨物自動車運送事業の許可をうけて、愛知県、岐阜県、三重県、及び静岡県を営業区域とし、主に食料品、日用品雑貨等、消費関連貨物の輸送を行っております
また、貨物運送取扱事業法に基づく第一、第二種利用運送事業の許可もうけております
現在、愛知県下に8支店、三重県下に1支店の拠点を持ち、倉庫業とともに総合的な物流サービスの一環として効率的な輸送サービスの提供を行っております
ロ 倉庫事業倉庫業法に基づく倉庫業の許可をうけて、愛知県下に2か所の営業倉庫と6か所の物流センター、三重県下に1か所の物流センターを持ち、貨物運送事業との連携により集荷・保管・流通加工・配送・回収までの一貫した総合物流サービスに努めております
ハ その他事業道路運送車両法に基づく自動車分解整備事業の認証をうけて、愛知県下に1か所の整備工場民間車検工場指定を持ち、自動車の車検、定期点検、一般修理を行っておりますほか、付帯して損害保険代理店事業を営んでおります
また、三好支店において太陽光発電事業を行っております
 また、子会社大宝興業株式会社はビルの賃貸を主たる業務としております

##### 回答
```json
{"事業内容":"貨物運送事業","具体的な説明":"食料品、日用品雑貨等の消費関連貨物の輸送に加え、第一・第二種利用運送事業も実施。"}
{"事業内容":"倉庫事業","具体的な説明":"営業倉庫と物流センターの運営。集荷・保管・流通加工・配送・回収までの一貫したサービスを提供"}
{"事業内容":"自動車整備事業","具体的な説明":"車検、定期点検、一般修理"}
{"事業内容":"損害保険代理店事業","具体的な説明":"なし"}
{"事業内容":"太陽光発電事業","具体的な説明":"なし"}
{"事業内容":"不動産賃貸事業","具体的な説明":"なし"}
```
以上の指示に従って、提供された文章から、事業内容を抽出して整理してください。
"""

prompt_ext_business = {
    "extsummary_instruction": """提供される文章はある会社の事業内容に関する文章の抜粋です。ここからわかる会社の主な事業内容を日本語で列挙してください。""",
    "extsummary_example": example_text,
    #### 注意事項
    "extsummary_constraints": [
        "できるだけ文章中の記載をそのまま引用してください。",
        "固有名詞は伏せてください。",
        "具体的な説明がある場合は、それも記載してください。ない場合は「なし」としてください。",
    ],
    "extsummary_output_formats": """#### 回答形式\n\nフォーマットは個別のjson形式で回答してください。\n\n{"事業内容":"(事業内容1)","具体的な説明":"(具体的な説明1)"}\n{"事業内容":"(事業内容2)","具体的な説明":"(具体的な説明2)"}""",
    #### 文章
    # ${}
}

sample_text = """3 【事業の内容】当社の主たる事業は物流業であります\nその事業は貨物運送事業、倉庫事業、その他事業に区分されますが、それぞれの事業内容は次のとおりであります\nイ 貨物運送事業貨物自動車運送事業法に基づく、一般貨物自動車運送事業の許可をうけて、愛知県、岐阜県、三重県、及び静岡県を営業区域とし、主に食料品、日用品雑貨等、消費関連貨物の輸送を行っております\nまた、貨物運送取扱事業法に基づく第一、第二種利用運送事業の許可もうけております\n現在、愛知県下に8支店、三重県下に1支店の拠点を持ち、倉庫業とともに総合的な物流サービスの一環として効率的な輸送サービスの提供を行っております\nロ 倉庫事業倉庫業法に基づく倉庫業の許可をうけて、愛知県下に2か所の営業倉庫と6か所の物流センター、三重県下に1か所の物流センターを持ち、貨物運送事業との連携により集荷・保管・流通加工・配送・回収までの一貫した総合物流サービスに努めております\nハ その他事業道路運送車両法に基づく自動車分解整備事業の認証をうけて、愛知県下に1か所の整備工場民間車検工場指定を持ち、自動車の車検、定期点検、一般修理を行っておりますほか、付帯して損害保険代理店事業を営んでおります\nまた、三好支店において太陽光発電事業を行っております\n また、子会社大宝興業株式会社はビルの賃貸を主たる業務としております"""
sys_p, usr_p = make_prompt_qag_prep(prompt_ext_business, sample_text)

print(sys_p)
print("===usr_p===")
print(usr_p)


example_text_risk = """
#### 例
##### 文章
4 【事業等のリスク】有価証券報告書に記載した事業の状況、経理の状況等に関する事項のうち投資者の判断に重要な影響を及ぼす可能性のある事項には、以下のようなものがあります
なお、文中の将来に関する事項は、当事業年度末現在において判断したものであります
1 公的規制について当社は、総合サービス物流企業として、貨物自動車運送事業、倉庫業等に関する各種法令の規制の適用を受けています
利益の確保と社会的責任の遂行によって、はじめて企業の発展が可能になるとの基本的スタンスで遵法経営を推進していますが、近年のトラック排ガス対策など環境関連規制の適用が強化されており、これらの事象が一層強化されれば、当社の業績及び財政状態に影響が及ぶ可能性があります
 2 取引関係の大幅な変動について当社は、企業物流の一括受託を主たる事業としており、顧客から物流業務を受託する際に、物流センター、荷役設備機器及び情報システム等について先行的に設備投資を実施することがあります
投資に際しては、綿密な事業収支計画を策定し、様々なリスクを予想し慎重に投資判断を行っておりますが、顧客の業績の急変や顧客との取引停止等により、投資資金の回収に支障が生じる可能性があります
従って、これらの事象は当社の将来の成長と収益性を低下させ、当社の業績及び財政状態に影響を及ぼす可能性があります
 3 燃料価格の変動について当社は、トラック輸送事業を主体とすることから、物流事業遂行にあたり燃料軽油の使用が不可欠になっています
安定的かつ適正価格で供給を受けていますが、世界の原油情勢の変動により燃料費が大幅に高騰し、輸配送コストが上昇する可能性があります
 4 物流料金の値下げについて当社の主要な取扱品は、一般の食品や日用品を基盤としております
この業界は厳しい競争に直面しており、商品の販売価格の低下傾向に伴い、物流コストも低く抑える動きが強くなっております
当社は、コスト削減に向けた運営体制の改革により、安定した利益率の確保に努めていますが、価格競争の更なる激化や長期化により、収益面を圧迫する可能性があります
従って、これらの事象は当社の業績及び財政状態に影響を及ぼす可能性があります

##### 回答
```json
{"リスク":"貨物自動車運送事業、倉庫業等に関する各種法令規制の適用","具体的な説明":"トラック排ガス対策など環境関連規制の強化"}
{"リスク":"取引関係の大幅な変動","具体的な説明":"物流業務受託時に先行的設備投資しているため、顧客の業績の急変や取引停止等による投資資金の回収リスクがある"}
{"リスク":"燃料価格の変動","具体的な説明":"世界の原油情勢の変動による燃料費の高騰による輸配送コストの上昇"}
{"リスク":"食品・日用品業界における価格競争","具体的な説明":"商品の販売価格の低下傾向に伴い、物流コストも低く抑える動き"}
```
以上の指示に従って、提供された文章から、事業内容を抽出して整理してください。
"""

prompt_ext_risk = {
    "extsummary_instruction": """提供される文章はある会社の事業のリスクに関する文章の抜粋です。ここに記載されているリスクを日本語で列挙してください。""",
    "extsummary_example": example_text_risk,
    #### 注意事項
    "extsummary_constraints": [
        "できるだけ文章中の記載をそのまま引用してください。",
        "固有名詞は伏せてください。",
        "具体的な説明がある場合は、それも記載してください。ない場合は「なし」としてください。",
    ],
    "extsummary_output_formats": """#### 回答形式\n\nフォーマットは個別のjson形式で回答してください。\n\n{"リスク":"(リスク1)","具体的な説明":"(具体的な説明1)"}\n{"リスク":"(リスク2)","具体的な説明":"(具体的な説明2)"}""",
    #### 文章
    # ${}
}
sys_p, usr_p = make_prompt_qag_prep(prompt_ext_risk, sample_text)

print(sys_p)
print("===usr_p===")
print(usr_p)


# Business
# %%
prompt_ext = {"business": prompt_ext_business, "risk": prompt_ext_risk}

# %%

# %% dataset

# filename=PROJDIR / "data/0_metadata/dataset_2407/response_tbl_rst_2407_v1012_chk2.csv"
# chk_df=pd.read_csv(filename)


# %%


TESTDIR = Path(PROJPATH) / "tests/20250115"
# model_name = "llama_3.1_70b"
# tag_name = "anly"
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
filename = (
    PROJPATH
    + "data/3_processed/dataset_2507/restatement/response_tbl_teisei_2507_v250803_with_year.pkl"
)
response_tbl = pd.read_pickle(filename)
response_tbl = response_tbl.query("year != 'not_found' and year != 'no_pre_file'")
response_tbl = response_tbl.rename(
    columns={
        "edinetCode": "response_edinetCode",
        "periodStart": "response_periodStart",
        "periodEnd": "response_periodEnd",
        "secCode": "response_secCode",
    },
)

conf_business = {
    "tag": "business",
    "include_tree_top_keyword": "OverviewOfCompanyHeading",
    "include_keyword_list": [
        "OverviewOfBusinessHeading",
        "DescriptionOfBusinessHeading",
        "DescriptionOfBusinessTextBlock",
        "OverviewOfAffiliatedEntitiesHeading",
        "OverviewOfAffiliatedEntitiesTextBlock",
    ],
    "exclude_keyword": "Employee",
    "keep_taxonomi_list": [
        "jpcrp_cor:OverviewOfBusinessHeading",
        "jpcrp_cor:DescriptionOfBusinessHeading",
        "jpcrp_cor:DescriptionOfBusinessTextBlock",
        "jpcrp_cor:OverviewOfAffiliatedEntitiesHeading",
        "jpcrp_cor:OverviewOfAffiliatedEntitiesTextBlock",
    ],
}

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
                conf=conf_business,
            )
            for itr_docID in response_tbl.index
        ],
    )
# %%


# %% save
import pickle

filename = (
    PROJPATH + "data/3_processed/dataset_2507/restatement/extracted_text_business.pkl"
)
with open(filename, "wb") as f:
    pickle.dump(results, f)

# %%
conf_risk = {
    "tag": "risk",
    "include_tree_top_keyword": "OverviewOfBusinessHeading",
    "include_keyword_list": ["BusinessRisk"],
    "exclude_keyword": [],
    "keep_taxonomi_list": [
        "jpcrp_cor:BusinessRisksHeading",
        "jpcrp_cor:BusinessRisksTextBlock",
        "jpcrp_cor:MaterialMattersRelatingToGoingConcernEtcBusinessRisksTextBlock",
    ],
}

n_parallel = 16
with tqdm_joblib(
    tqdm(desc="My calculation", total=len(response_tbl.index)),
) as progress_bar:
    results_risk = joblib.Parallel(n_jobs=n_parallel, verbose=0)(
        [
            joblib.delayed(preproc_fs)(
                zip_file_str=get_zipdir2(itr_docID),
                itr_docID=itr_docID,
                response_tbl=response_tbl,
                role_keyward_list=fs_dict["report"],
                conf=conf_risk,
            )
            for itr_docID in response_tbl.index
        ],
    )
# save
filename = (
    PROJPATH + "data/3_processed/dataset_2507/restatement/extracted_text_risk.pkl"
)
with open(filename, "wb") as f:
    pickle.dump(results_risk, f)

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
        sys_prompt, usr_prompt = make_prompt_qag_prep(prompt_ext["business"], text)
        itr_index_str = str(itr_docID) + "_" + "business" + "_" + str(itr_num)
        batch_inf_file_generator_obj.insert_inf_list_prompt(
            sys_prompt,
            usr_prompt,
            itr_index_str,
            max_tokens=2048,
            model_name=model_name,
        )

out_filename = (
    PROJDIR
    / "data/3_processed/dataset_2507/restatement/llm_proc"
    / ("batch_extract_text_" + model_name + "_business.jsonl")
)
batch_inf_file_generator_obj.export_list(out_filename)
batch_inf_file_generator_obj.print_sample()


rst_list = []
batch_inf_file_generator_obj = batch_inf_file_generator(
    model_name=model_name,
)


for result_list in results_risk:
    if len(result_list) == 0:
        continue
    for itr_num, text_dict in enumerate(result_list):
        text = text_dict["text"]

        itr_docID = text_dict["docID"]
        sys_prompt, usr_prompt = make_prompt_qag_prep(prompt_ext["risk"], text)
        itr_index_str = str(itr_docID) + "_" + "risk" + "_" + str(itr_num)
        batch_inf_file_generator_obj.insert_inf_list_prompt(
            sys_prompt,
            usr_prompt,
            itr_index_str,
            max_tokens=2048,
            model_name=model_name,
        )

out_filename = (
    PROJDIR
    / "data/3_processed/dataset_2507/restatement/llm_proc"
    / ("batch_extract_text_" + model_name + "_risk.jsonl")
)
batch_inf_file_generator_obj.export_list(out_filename)
batch_inf_file_generator_obj.print_sample()


# %% memo ########################################################################
model_name = "llama_3.1_8b"


batch_inf_file_generator_obj = batch_inf_file_generator(
    model_name=model_name,
)
for itr_docid in tqdm(response_tbl.index):
    # text block
    data_path = response_tbl.loc[itr_docid, :].dataset

    filepath = (
        PROJDIR / "data" / "2_intermediate" / ("data_pool_" + data_path) / itr_docid
    )
    text_block_file = filepath / "text_block.json"
    text_block_df = pd.read_json(text_block_file)

    #    for tag_name in ['business','risk']:
    for tag_name in ["business"]:
        if len(text_block_df.query("tag==@tag_name")) > 0:
            sample_text_all = text_block_df.query("tag==@tag_name").iloc[0, :].text
            sample_text_list = multi_stage_chunk(sample_text_all, max_text_length=20000)
            for itr_num, text in enumerate(sample_text_list):
                sys_prompt, usr_prompt = make_prompt_qag_prep(
                    prompt_ext[tag_name],
                    text,
                )
                itr_index_str = (
                    str(itr_docid) + "_" + str(tag_name) + "_" + str(itr_num)
                )
                batch_inf_file_generator_obj.insert_inf_list_prompt(
                    sys_prompt,
                    usr_prompt,
                    itr_index_str,
                    max_tokens=4096,
                    model_name=model_name,
                )

out_filename = (
    PROJDIR
    / "data/3_processed/dataset_2310/llm_proc"
    / ("batch_extract_text_" + model_name + "_bussiness.jsonl")
)
batch_inf_file_generator_obj.export_list(out_filename)
batch_inf_file_generator_obj.print_sample()


# %%

batch_inf_file_generator_obj = batch_inf_file_generator(
    model_name=model_name,
)
accounting_standard = []
for itr_docid in tqdm(response_tbl.index):
    # text block
    data_path = response_tbl.loc[itr_docid, :].dataset

    filepath = (
        PROJDIR / "data" / "2_intermediate" / ("data_pool_" + data_path) / itr_docid
    )
    text_block_file = filepath / "text_block.json"
    text_block_df = pd.read_json(text_block_file)

    metadata_file = filepath / "XBRL" / "PublicDoc" / "log_dict.json"
    with open(metadata_file) as f:
        metadata_file_dict = json.load(f)
    accounting_standard.append(
        {
            "docid": itr_docid,
            "AccountingStandardsDEI": metadata_file_dict["AccountingStandardsDEI"],
        },
    )

    #    for tag_name in ['business','risk']:
    for tag_name in ["risk"]:
        if len(text_block_df.query("tag==@tag_name")) > 0:
            sample_text_all = text_block_df.query("tag==@tag_name").iloc[0, :].text
            sample_text_list = multi_stage_chunk(sample_text_all, max_text_length=20000)
            for itr_num, text in enumerate(sample_text_list):
                sys_prompt, usr_prompt = make_prompt_qag_prep(
                    prompt_ext[tag_name],
                    text,
                )
                itr_index_str = (
                    str(itr_docid) + "_" + str(tag_name) + "_" + str(itr_num)
                )
                batch_inf_file_generator_obj.insert_inf_list_prompt(
                    sys_prompt,
                    usr_prompt,
                    itr_index_str,
                    max_tokens=2048,
                    model_name=model_name,
                )

out_filename = (
    PROJDIR
    / "data/3_processed/dataset_2310/llm_proc"
    / ("batch_extract_text_" + model_name + "_col_risk.jsonl")
)
batch_inf_file_generator_obj.export_list(out_filename)
batch_inf_file_generator_obj.print_sample()

# %%
# %%
# Business　2


# %% あんまり具体的に書いていないからスルー

example_text_risk2 = """
#### 例
##### 文章
4 【事業等のリスク】有価証券報告書に記載した事業の状況、経理の状況等に関する事項のうち投資者の判断に重要な影響を及ぼす可能性のある事項には、以下のようなものがあります
なお、文中の将来に関する事項は、当事業年度末現在において判断したものであります
1 公的規制について当社は、総合サービス物流企業として、貨物自動車運送事業、倉庫業等に関する各種法令の規制の適用を受けています
利益の確保と社会的責任の遂行によって、はじめて企業の発展が可能になるとの基本的スタンスで遵法経営を推進していますが、近年のトラック排ガス対策など環境関連規制の適用が強化されており、これらの事象が一層強化されれば、当社の業績及び財政状態に影響が及ぶ可能性があります
 2 取引関係の大幅な変動について当社は、企業物流の一括受託を主たる事業としており、顧客から物流業務を受託する際に、物流センター、荷役設備機器及び情報システム等について先行的に設備投資を実施することがあります
投資に際しては、綿密な事業収支計画を策定し、様々なリスクを予想し慎重に投資判断を行っておりますが、顧客の業績の急変や顧客との取引停止等により、投資資金の回収に支障が生じる可能性があります
従って、これらの事象は当社の将来の成長と収益性を低下させ、当社の業績及び財政状態に影響を及ぼす可能性があります
 3 燃料価格の変動について当社は、トラック輸送事業を主体とすることから、物流事業遂行にあたり燃料軽油の使用が不可欠になっています
安定的かつ適正価格で供給を受けていますが、世界の原油情勢の変動により燃料費が大幅に高騰し、輸配送コストが上昇する可能性があります
 4 物流料金の値下げについて当社の主要な取扱品は、一般の食品や日用品を基盤としております
この業界は厳しい競争に直面しており、商品の販売価格の低下傾向に伴い、物流コストも低く抑える動きが強くなっております
当社は、コスト削減に向けた運営体制の改革により、安定した利益率の確保に努めていますが、価格競争の更なる激化や長期化により、収益面を圧迫する可能性があります
従って、これらの事象は当社の業績及び財政状態に影響を及ぼす可能性があります

##### 回答
```json
{"リスク":"貨物自動車運送事業、倉庫業等に関する各種法令規制の適用","会社の方針":"利益の確保と社会的責任の遂行によって、はじめて企業の発展が可能になるとの基本的スタンスで遵法経営を推進","業績及び財政状態への影響":"近年のトラック排ガス対策など環境関連規制の適用が強化"}
{"リスク":"取引関係の大幅な変動","会社の方針":"投資に際しては、綿密な事業収支計画を策定し、様々なリスクを予想し慎重に投資判断を行っている","業績及び財政状態への影響":"顧客の業績の急変や顧客との取引停止等により、投資資金の回収に支障が生じる可能性"}
{"リスク":"燃料価格の変動","会社の方針":"記載なし","業績及び財政状態への影響":"安定的かつ適正価格で供給を受けているが、世界の原油情勢の変動により燃料費が大幅に高騰し、輸配送コストが上昇する可能性がある"}
{"リスク":"食品・日用品業界における価格競争","会社の方針":"コスト削減に向けた運営体制の改革により、安定した利益率の確保に努めている","業績及び財政状態への影響":"価格競争の更なる激化や長期化により、収益面を圧迫する可能性"}
```
以上の指示に従って、提供された文章から、事業内容を抽出して整理してください。
"""

prompt_ext_risk2 = {
    "extsummary_instruction": """提供される文章はある会社の事業のリスクに関する文章の抜粋です。ここに記載されているリスク、会社の方針、業績及び財政状態の影響を日本語で抽出し列挙してください。""",
    "extsummary_example": example_text_risk2,
    #### 注意事項
    "extsummary_constraints": [
        "できるだけ文章中の記載をそのまま引用してください。",
        "固有名詞は伏せてください。",
        "該当する記載がない場合は「記載なし」としてください。",
    ],
    "extsummary_output_formats": """#### 回答形式\n\nフォーマットは個別のjson形式で回答してください。\n\n{"リスク":"(リスク1)","会社の方針":"(会社の方針1)","業績及び財政状態への影響":"(業績及び財政状態への影響1)"}\n{"リスク":"(リスク2)","会社の方針":"(会社の方針2)","業績及び財政状態への影響":"(業績及び財政状態への影響2)"}""",
    #### 文章
    # ${}
}
# Business
# %%
prompt_ext = {"risk": prompt_ext_risk2}


# %% dataset

filename = PROJDIR / "data/0_metadata/dataset_2407/response_tbl_rst_2407_v1012_chk2.csv"
chk_df = pd.read_csv(filename)
# %%
failure_docid_set = set(chk_df.docid)
filename = PROJDIR / "data/0_metadata/dataset_2407/response_tbl_rst_2407_v1012.pkl"
response_tbl = pd.read_pickle(filename)  # .head()
print(len(response_tbl))  # 43994
response_tbl = response_tbl.query("index not in @failure_docid_set")
print(len(response_tbl))  # 43977

# %%
model_name = "llama_3.1_8b"

batch_inf_file_generator_obj = batch_inf_file_generator(
    model_name=model_name,
)
# accounting_standard=[]
for itr_docid in tqdm(response_tbl.index):
    # text block
    data_path = response_tbl.loc[itr_docid, :].dataset

    filepath = (
        PROJDIR / "data" / "2_intermediate" / ("data_pool_" + data_path) / itr_docid
    )
    text_block_file = filepath / "text_block.json"
    text_block_df = pd.read_json(text_block_file)

    # metadata_file=filepath /"XBRL"/ "PublicDoc" / "log_dict.json"
    # with open(metadata_file) as f:
    #    metadata_file_dict = json.load(f)
    # accounting_standard.append({"docid":itr_docid,"AccountingStandardsDEI":metadata_file_dict['AccountingStandardsDEI']})

    for tag_name in ["risk"]:
        if len(text_block_df.query("tag==@tag_name")) > 0:
            sample_text_all = text_block_df.query("tag==@tag_name").iloc[0, :].text
            sample_text_list = multi_stage_chunk(sample_text_all, max_text_length=20000)
            for itr_num, text in enumerate(sample_text_list):
                sys_prompt, usr_prompt = make_prompt_qag_prep(
                    prompt_ext[tag_name],
                    text,
                )
                itr_index_str = (
                    str(itr_docid) + "_" + str(tag_name) + "_" + str(itr_num)
                )
                batch_inf_file_generator_obj.insert_inf_list_prompt(
                    sys_prompt,
                    usr_prompt,
                    itr_index_str,
                    max_tokens=4096,
                    model_name=model_name,
                )

out_filename = (
    PROJDIR
    / "data/3_processed/dataset_2310/llm_proc"
    / ("batch_extract_text_" + model_name + "_risk_act.jsonl")
)
batch_inf_file_generator_obj.export_list(out_filename)
batch_inf_file_generator_obj.print_sample()

# %%
itr_docid = response_tbl.index[-100]
data_path = response_tbl.loc[itr_docid, :].dataset

filepath = PROJDIR / "data" / "2_intermediate" / ("data_pool_" + data_path) / itr_docid
text_block_file = filepath / "text_block.json"
text_block_df = pd.read_json(text_block_file)
tag_name = "anly"
print(text_block_df.query("tag==@tag_name").iloc[0, :].text)

# %%
temp_path_str = str(
    PROJDIR
    / "data"
    / "2_intermediate"
    / (f"data_pool_{response_tbl.loc[itr_docID, 'dataset']}")
    / itr_docID,
)
filename = temp_path_str + "/fs_tbl.pkl"
fs_tbl = FsDataDf(pd.read_pickle(filename))

# %%
fs_tbl.order

# %%
