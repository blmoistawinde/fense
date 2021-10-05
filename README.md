# FENSE

The metric, **F**luency **EN**hanced **S**entence-bert **E**valuation (FENSE), for audio caption evaluation, proposed in the paper "Can Audio Captions Be Evaluated with Image Caption Metrics?"

The `main` branch contains an easy-to-use interface for fast evaluation of an audio captioning system. To get the dataset and the code to reproduce, please refer to the [experiment-code](https://github.com/blmoistawinde/fense/tree/experiment-code) branch.

## Installation

Clone the reporsitory and pip install it.

```bash
git clone https://github.com/blmoistawinde/fense.git
cd fense
pip install -e .
```

## Usage

### Single Sentence
To get the detailed scores of each component for a single sentence.

```python
from fense.evaluator import Evaluator

print("----Using tiny models----")
evaluator = Evaluator(device='cpu', sbert_model='paraphrase-MiniLM-L6-v2', echecker_model='echecker_clotho_audiocaps_tiny')

eval_cap = "An engine in idling and a man is speaking and then"
ref_cap = "A machine makes stitching sounds while people are talking in the background"

score, error_prob, penalized_score = evaluator.sentence_score(eval_cap, [ref_cap], return_error_prob=True)

print("Cand:", eval_cap)
print("Ref:", ref_cap)
print(f"SBERT sim: {score:.4f}, Error Prob: {error_prob:.4f}, Penalized score: {penalized_score:.4f}")
```

### System Score

To get a system's overall score on a dataset by averaging sentence-level FENSE, you can use `eval_system.py`, with your system outputs prepared in the format like `test_data/audiocaps_cands.csv` or `test_data/clotho_cands.csv` .

For AudioCaps test set:

```bash
python eval_system.py --device cuda --dataset audiocaps --cands_dir ./test_data/audiocaps_cands.csv
```

For Clotho Eval set:

```bash
python eval_system.py --device cuda --dataset clotho --cands_dir ./test_data/clotho_cands.csv
```

## Reference

TODO
