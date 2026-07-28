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
python AARL/train.py
```

AARL/cfg contains config files for training settings (hyperparameters) for the paper.

