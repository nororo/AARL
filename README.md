# AARL
Repository for "Accounting-Aware Representation Learning for Company Similarity Assessment"


### 1. Dataset


```
edinet_dataset/
├── eval/
|   ├── docid_tbl_eval_300edinetcodes.csv #
|   ├── sector_eval_268edinetcodes.csv #
|
├── train/
|   ├──docid_tbl_all_140101_250331.pkl # all doc id list from 2014-01-01 to 2025-03-31
```

This dataset is created from the original data from EDINET. Therefore, term of use of original data from EDINET is applied. https://disclosure2.edinet-fsa.go.jp/week0010.aspx#

For text data in edinet/dataset/train/, in addition to the term of use above ,since the text in the dataset were extracted from original documents using Llama-3.1 model, term of use of Meta Llama model is also applied. https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE

**full dataset is too large to be uploaded to the repository.**
Google Drive link will be provided, when the paper is accepted.
Sample from the dataset are available in the edinet_dataset/train/sample directory.


### 2. Train AARL model
Model definition is in the AARL/multi_account_clip_common directory.
You cna train AARL model by using the endpoint script:
```
python AARL/train.py --cfg AARL/cfg/aarl_config.yaml
```

AARL/cfg contains config files for training settings (hyperparameters) for the paper.


### Data collection
This scripts for making edinet_dataset. (You do not need to run this scripts, since the dataset is already collected.)

#### Financial statement account values
Data collection is done by using our other open source project repository:

https://anonymous.4open.science/r/edinet_xbrl_prep-35E8/

This repository is used to collect XBRL data from EDINET API and preprocess the data, resulting in the one big table.

#### Text data

Text data is extracted from the XBRL documents the repository collected, by set role text to the role textual part of the annual securities report.

The following code is an example of how to extract text data from the XBRL documents, after you have collected the XBRL documents.
```
role_text = "http://disclosure.edinet-fsa.go.jp/role/jpcrp/rol_CabinetOfficeOrdinanceOnDisclosureOfCorporateInformationEtcFormNo3AnnualSecuritiesReport"

fs_tbl_df:FsDataDf = get_fs_tbl(
    account_list_common_obj=account_list_common_obj_2024,
    docid=docid, # document id (filename of downloaded zip file removed extension)
    zip_file_str=str(DATA_PATH / "raw/xbrl_doc" / (docid + ".zip")),# downloaded zip file
    temp_path_str=str(DATA_PATH / "raw/xbrl_doc_ext" / docid), # for temporal file
    role_keyward_list=[role_text],
)
```

