# AARL
Repository for "Accounting-Aware Representation Learning for Company Similarity Assessment"


### Dataset

```
edinet_dataset/
├── eval/
|   ├── docid_tbl_eval_300edinetcodes.csv #
|   ├── sector_eval_268edinetcodes.csv #
|
├── train/
|   ├──
```

### 3. Train AARL model
Model definition is in the AARL/multi_account_clip_common directory.
You cna train AARL model by using the endpoint script:
```
python AARL/train.py
```

AARL/cfg contains config files for training settings for the paper.

### 4. Inference shopping embedding features

Run the `slp/inference_shopping_embedding.py` script to inference the shopping embedding features.
```
python slp/inference_shopping_embedding.py
```

This will create the following files in the data directory:
```
data/
├── features.parquet
```

### 5. Generate substitute pairs

Run the `process_data.py` script to generate the substitute pairs.
```
python process_data.py
```

This will create substitute_pairs files in the data/method_name directory:
```
data/
├── {method_name}
│   ├── substitute_pairs.parquet
```

