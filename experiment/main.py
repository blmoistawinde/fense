import json
import numpy as np
import pandas as pd
import json 
import torch
from tqdm import tqdm
import sys
sys.path.append('../caption-evaluation-tools')
from sentence_transformers import SentenceTransformer, CrossEncoder
from bert_score import BERTScorer
from bleurt import score as bleurt_score
from eval_metrics import evaluate_metrics_from_lists
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dataloader import get_former, get_latter

model_sb = SentenceTransformer('paraphrase-TinyBERT-L6-v2', device='cuda:1')
model_sb.eval()

scorer_bs = BERTScorer(model_type='ramsrigouthamg/t5_paraphraser', lang="en", rescale_with_baseline=False, device='cuda:3')

scorer_brt = bleurt_score.BleurtScorer()

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def cosine_similarity(input, target):
    from torch.nn import CosineSimilarity
    cos = CosineSimilarity(dim=0, eps=1e-6)
    return cos(input, target).item()
    
def get_text_score(all_preds_text, all_refs_text, method='sentence-bert', average=True):
    N = len(all_preds_text)
    K = len(all_refs_text[0])
    all_preds_text = np.array(all_preds_text, dtype=str)
    all_refs_text = np.array(all_refs_text, dtype=str)

    score = torch.zeros((N, K))
    if method == 'sentence-bert':
        preds_sb = torch.Tensor(model_sb.encode(all_preds_text))
        refs_sb = torch.Tensor(np.array([model_sb.encode(x) for x in all_refs_text]))
        # refs_sb = refs_sb.mean(dim=1)
        for i in range(K):
            score[:,i] = torch.Tensor([cosine_similarity(input, target) for input, target in zip(preds_sb, refs_sb[:,i])])
    elif method == 'bert-score':
        for i in range(K):
            P, R, F1 = scorer_bs.score(all_preds_text.tolist(), all_refs_text[:,i].tolist())
            score[:,i] = F1
    elif method == 'bleurt':
        for i in range(K):
            scores = scorer_brt.score(references=all_refs_text[:,i], candidates=all_preds_text)
            score[:,i] = torch.Tensor(scores).sigmoid()

    score = score.mean(dim=1) if average else score.max(dim=1)[0]

    return score

def get_accuracy(machine_score, human_score, threshold=0):
    cnt = 0
    # threshold = 0.001*np.average([abs(t) for t in machine_score])
    # threshold = 1e-6
    N = np.sum([x!=0 for x in human_score]) if threshold==0 else len(human_score)
    for i, (ms, hs) in enumerate(zip(machine_score, human_score)):
        if ms*hs > 0 or abs(ms-hs) < threshold:
            cnt += 1
    return cnt / N

def print_accuracy(machine_score, human_score):
    results = []
    for i, facet in enumerate(['HC', 'HI', 'HM', 'MM']):
        if facet != 'MM':
            sub_score = machine_score[i*250:(i+1)*250]
            sub_truth = human_score[i*250:(i+1)*250]
        else:
            sub_score = machine_score[i*250:]
            sub_truth = human_score[i*250:]
        acc = get_accuracy(sub_score, sub_truth)
        results.append(round(acc*100, 1))
        print(facet,  "%.1f" % (acc*100))
    acc = get_accuracy(machine_score, human_score)
    results.append(round(acc*100, 1))
    print("total acc: %.1f" % (acc*100))
    return results

if __name__ == '__main__':
    for dataset in ['audiocaps', 'clotho']:
        score, score0, score1 = {}, {}, {}

        mm_score, mm_score0, mm_score1 = {}, {}, {}

        hh_preds_text0, hh_preds_text1, hh_refs_text0, hh_refs_text1, hh_human_truth = get_former(dataset)
        mm_preds_text0, mm_preds_text1, mm_refs_text, mm_human_truth = get_latter(dataset)

        for metric in ['sentence-bert', 'bleurt', 'bert-score']:
            score0[metric] = get_text_score(hh_preds_text0, hh_refs_text0, metric)
            score1[metric] = get_text_score(hh_preds_text1, hh_refs_text1, metric)

            mm_score0[metric] = get_text_score(mm_preds_text0, mm_refs_text, metric)
            mm_score1[metric] = get_text_score(mm_preds_text1, mm_refs_text, metric)

        total_score0, total_score1, total_score = {}, {}, {}
        for metric in score0:
            total_score0[metric] = torch.cat([score0[metric], mm_score0[metric]])
            total_score1[metric] = torch.cat([score1[metric], mm_score1[metric]])
            total_score[metric] = total_score0[metric] - total_score1[metric]
        total_human_truth = hh_human_truth + mm_human_truth

        metrics0, per_file_metrics0 = evaluate_metrics_from_lists(hh_preds_text0, hh_refs_text0)
        metrics1, per_file_metrics1 = evaluate_metrics_from_lists(hh_preds_text1, hh_refs_text1)

        # expand the references (choose 4 from 5)
        mm_preds_text0_exp = [x for x in mm_preds_text0 for i in range(5)]
        mm_preds_text1_exp = [x for x in mm_preds_text1 for i in range(5)]
        mm_refs_text_exp = []
        for refs in mm_refs_text:
            for i in range(5):
                mm_refs_text_exp.append([v for k,v in enumerate(refs) if k%5!=i])

        mm_metrics0, mm_per_file_metrics0 = evaluate_metrics_from_lists(mm_preds_text0_exp, mm_refs_text_exp)
        mm_metrics1, mm_per_file_metrics1 = evaluate_metrics_from_lists(mm_preds_text1_exp, mm_refs_text_exp)

        def get_score_list(per_file_metric, metric):
            if metric == 'SPICE':
                return [v[metric]['All']['f'] for k,v in per_file_metric.items()]
            else:
                return [v[metric] for k,v in per_file_metric.items()]

        def shrink(arr, repeat=5):
            return np.array(arr).reshape(-1, repeat).mean(axis=1).tolist()

        baseline_list = ['Bleu_1','Bleu_2','Bleu_3','Bleu_4','METEOR','ROUGE_L','CIDEr','SPICE','SPIDEr']
        for metric in baseline_list:
            total_score0[metric] = torch.Tensor(get_score_list(per_file_metrics0, metric) + shrink(get_score_list(mm_per_file_metrics0, metric)))
            total_score1[metric] = torch.Tensor(get_score_list(per_file_metrics1, metric) + shrink(get_score_list(mm_per_file_metrics1, metric)))
            total_score[metric] = total_score0[metric] - total_score1[metric]

        results = []
        for metric in total_score:
            print(metric)
            tmp = print_accuracy(total_score[metric], total_human_truth)
            results.append(tmp)

        df = pd.DataFrame(results, columns=['HC', 'HI', 'HM', 'MM', 'total'])
        df.index = [x for x in total_score]
        df.to_csv('results_{}.csv'.format(dataset))

        ##############################################################################
        # Error detector applied
        ##############################################################################

        # load pre-computed ndarray 
        probs0 = np.load('../bert_for_fluency/cache/probs0_alltrain_{}.npy'.format(dataset))
        probs1 = np.load('../bert_for_fluency/cache/probs1_alltrain_{}.npy'.format(dataset))

        score_penalty = {}
        thres = 0.9
        coef = 0.9

        for method in total_score:
            score_penalty[method] = [s1-s1*coef*(p1>thres)-(s2-s2*coef*(p2>thres)) for s1,s2,p1,p2 in zip(total_score0[method],total_score1[method],probs0[:,-1],probs1[:,-1])]

        results = []
        for method in score_penalty:
            print(method)
            tmp = print_accuracy(score_penalty[method], total_human_truth)
            results.append(tmp)

        df = pd.DataFrame(results, columns=['HC', 'HI', 'HM', 'MM', 'total'])
        df.index = [x for x in total_score]
        df.to_csv('fluency_{}.csv'.format(dataset))