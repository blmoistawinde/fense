import json
import numpy as np
from tqdm import tqdm
import pandas as pd

with open('../dataset/audiocaps_eval.json') as f:
    anno_audiocaps = json.load(f)
with open('../dataset/clotho_eval.json') as f:
    anno_clotho = json.load(f)

def expand_refs(all_refs_text):
    for idx,refs in enumerate(all_refs_text):
        i = 0
        while len(refs) < 4:
            refs.append(refs[i])
            i += 1
        all_refs_text[idx] = refs
    return all_refs_text
    
def swap_position(arr):
    ret = []
    for i in range(3):
        ret += [v for k,v in enumerate(arr) if k%3==i]
    return ret

def get_former(dataset='clotho'):
    assert dataset in {'audiocaps', 'clotho'}
    hh_human_truth = []
    hh_preds_text0 = []
    hh_preds_text1 = []
    hh_refs_text0 = []
    hh_refs_text1 = []

    anno = anno_clotho if dataset == 'clotho' else anno_audiocaps
    
    for audio in tqdm(anno):
        captions = audio["references"]
        for facet in ["HC", "HI", "HM"]:
            try:
                truth = np.sum([x for x in audio[facet][-1]])
                hh_preds_text0.append(audio[facet][0])
                hh_preds_text1.append(audio[facet][1])
                hh_human_truth.append(truth)
                if facet == "HC":
                    hh_refs_text0.append([x for x in captions if x != audio[facet][0]])
                    hh_refs_text1.append([x for x in captions if x != audio[facet][1]])
                elif facet == "HI" or facet == "HM":
                    hh_refs_text0.append([x for x in captions if x != audio[facet][0]])
                    hh_refs_text1.append([x for x in captions if x != audio[facet][0]])   
            except:
                continue

    hh_refs_text0 = expand_refs(hh_refs_text0)
    hh_refs_text1 = expand_refs(hh_refs_text1)

    hh_preds_text0 = swap_position(hh_preds_text0)
    hh_preds_text1 = swap_position(hh_preds_text1)
    hh_refs_text0 = swap_position(hh_refs_text0)
    hh_refs_text1 = swap_position(hh_refs_text1)
    hh_human_truth = swap_position(hh_human_truth)
    return hh_preds_text0, hh_preds_text1, hh_refs_text0, hh_refs_text1, hh_human_truth

def get_latter(dataset='clotho'):
    assert dataset in {'audiocaps', 'clotho'}
    mm_human_truth = []
    mm_preds_text0 = []
    mm_preds_text1 = []
    mm_refs_text = []

    anno = anno_clotho if dataset == 'clotho' else anno_audiocaps

    for audio in tqdm(anno):
        captions = audio["references"]
        for facet in ["MM_1", "MM_2", "MM_3", "MM_4", "MM_5"]:
            try:
                truth = np.sum([x for x in audio[facet][-1]])
                mm_preds_text0.append(audio[facet][0])
                mm_preds_text1.append(audio[facet][1])
                mm_refs_text.append(captions)
                mm_human_truth.append(truth)
            except:
                continue

    return mm_preds_text0, mm_preds_text1, mm_refs_text, mm_human_truth

if __name__ == '__main__':
    all_preds_text0, all_preds_text1, all_refs_text0, all_refs_text1, all_human_truth = get_former('audiocaps')
    mm_preds_text0, mm_preds_text1, mm_refs_text, mm_human_truth = get_latter('audiocaps')

    print(len(all_human_truth))

    # import pdb; pdb.set_trace()
    print(np.array(all_refs_text0, dtype=str).shape)
    print(np.array(mm_preds_text0, dtype=str).shape)

    # np.save('total_preds_audiocaps0.npy', all_preds_text0 + mm_preds_text0)
    # np.save('total_preds_audiocaps1.npy', all_preds_text1 + mm_preds_text1)
