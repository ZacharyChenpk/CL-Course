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

3. ```bash
chmod +x run_model.sh
./run_model.sh
```