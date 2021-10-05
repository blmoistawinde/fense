# Experiment code and datasets 
To reproduce the experiment results in our paper "Can Audio Captions Be Evaluated with Image Caption Metrics?", please clone this branch and install the requirements:
## Installation
```
git clone -b experiment-code https://github.com/blmoistawinde/fense.git
cd fense
pip install -e .
```

Please read the README.md in folder `caption-evaluation-tools` and run the script in `caption-evaluation-tools/coco_caption/get_stanford_models.sh` to download the required models as well. 
## Run
Since we have cached the inference results of *Error Detector* in `bert_for_fluency` folder, you can simply run
```
cd experiment
python main.py
```
The experiment results would be written into `results_[dataset_name].csv` and `fluency_[dataset_name].csv`, in which `dataset_name` refers to either audiocaps or clotho. 
## Dataset 
In this paper we've carefully annotated two benchmark datasets for audio caption evaluation, *AudioCaps-Eval* and *Clotho-Eval*. They should be seen in `dataset` folder.  

We also provide Fluency Issue annotations on a sample of 723 machined generated captions on AudioCaps at `dataset/fluency_annotations_723.csv`