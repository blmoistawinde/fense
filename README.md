# FENSE

The metric, **F**luency **EN**hanced **S**entence-bert **E**valuation (FENSE), for audio caption evaluation, proposed in the paper ["Can Audio Captions Be Evaluated with Image Caption Metrics?"](https://arxiv.org/abs/2110.04684)

The `main` branch contains an easy-to-use interface for fast evaluation of an audio captioning system.

Online demo avaliable at https://share.streamlit.io/blmoistawinde/fense/main/streamlit_demo/app.py .

To get the dataset (AudioCaps-Eval and Clotho-Eval) and the code to reproduce, please refer to the [experiment-code](https://github.com/blmoistawinde/fense/tree/experiment-code) branch.

## Installation

Clone the repository and pip install it.

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

## Performance Benchmark

We benchmark the performance of FENSE with different choices of SBERT model and Error Detector on the two benchmark dataset AudioCaps-Eval and Clotho-Eval. (*) is the combination reported in paper.

AudioCaps-Eval

| SBERT | echecker | HC   | HI   | HM   | MM   | total  |
|-------|-------|------|------|------|------|--------|
| paraphrase-MiniLM-L6-v2 |  none     | 62.1 | 98.8 | 93.7 | 75.4 | 80.4   |
| paraphrase-MiniLM-L6-v2 | tiny  | 57.6 | 94.7 | 89.5 | 82.6 | 82.3   |
| paraphrase-MiniLM-L6-v2 | base  | 62.6 | 98   | 82.5 | 85.4 | 85.5   |
| paraphrase-TinyBERT-L6-v2 | none    | 64   | 99.2 | 92.5 | 73.6 | 79.6   |
| paraphrase-TinyBERT-L6-v2 | tiny  | 58.6 | 95.1 | 88.3 | 82.2 | 82.1   |
| paraphrase-TinyBERT-L6-v2 | base  | 64.5 | 98.4 | 91.6 | 84.6 | 85.3(*)  |
| paraphrase-mpnet-base-v2  | none  | 63.1 | 98.8 | 94.1 | 74.1 | 80.1   |
| paraphrase-mpnet-base-v2 | tiny  | 58.1 | 94.3 | 90   | 83.2 | 82.7   |
| paraphrase-mpnet-base-v2 | base  | 63.5 | 98   | 92.5 | 85.9 | 85.9   |


Clotho-Eval

| SBERT | echecker | HC   | HI   | HM   | MM   | total  |
|-------|-------|------|------|------|------|--------|
| paraphrase-MiniLM-L6-v2 | none    | 59.5 | 95.1 | 76.3 | 66.2 | 71.3   |
| paraphrase-MiniLM-L6-v2 | tiny  | 56.7 | 90.6 | 79.3 | 70.9 | 73.3   |
| paraphrase-MiniLM-L6-v2 | base  | 60   | 94.3 | 80.6 | 72.3 | 75.3   |
| paraphrase-TinyBERT-L6-v2 | none  | 60   | 95.5 | 75.9 | 66.9 | 71.8   |
| paraphrase-TinyBERT-L6-v2 | tiny  | 59   | 93   | 79.7 | 71.5 | 74.4   |
| paraphrase-TinyBERT-L6-v2 | base  | 60.5 | 94.7 | 80.2 | 72.8 | 75.7(*)   |
| paraphrase-mpnet-base-v2  | none  | 56.2 | 96.3 | 77.6 | 65.2 | 70.7   |
| paraphrase-mpnet-base-v2 | tiny  | 54.8 | 91.8 | 80.6 | 70.1 | 73     |
| paraphrase-mpnet-base-v2 | base  | 57.1 | 95.5 | 81.9 | 71.6 | 74.9   |

## Reference

If you use FENSE in your research, please cite:

```
@misc{zhou2021audio,
      title={Can Audio Captions Be Evaluated with Image Caption Metrics?}, 
      author={Zelin Zhou and Zhiling Zhang and Xuenan Xu and Zeyu Xie and Mengyue Wu and Kenny Q. Zhu},
      year={2021},
      eprint={2110.04684},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
