# CL-Course

## Dependencies

```bash
pip install -r requirements.txt
```

## Getting Start

1. Modifying the module importing part of main.py Line 52 (Optional)
```python
from model_allcat_tag import unwrapped_preprocess_function, MyModule, DataCollatorForMultipleChoice, MyTokenizer, MyOptimizer
```

2. Modifying the parameters and available CUDA device in run_model.sh (Optional)

3. 
```bash
chmod +x run_model.sh
./run_model.sh
```

## Files Structure
.
├── CCPM-data ### Original and split data
│   ├── split_test.jsonl
│   ├── split_valid.jsonl
│   ├── test_public.jsonl
│   ├── train.jsonl
│   └── valid.jsonl
├── data_split.py ### code for splitting valid.jsonl
├── main.py ### entry of all models except PLM-Match
├── main_sim.py ### entry of PLM-Match model
├── model_allcat.py ### PLM-All
├── model_allcat_tag.py ### PLM-All-Tag
├── model_dual_tag_cat.py ### PLM-DualTag
├── model.py ### PLM-CLS
├── model_sim.py ### PLM-Match
├── model_with_tag.py ### PLM-Tag
├── README.md
├── requirements.txt
├── run_model.sh
