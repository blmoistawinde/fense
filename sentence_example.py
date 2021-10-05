from fense.evaluator import Evaluator

print("----Using tiny models----")
evaluator = Evaluator(device='cpu', sbert_model='paraphrase-MiniLM-L6-v2', echecker_model='echecker_clotho_audiocaps_tiny')

eval_cap = "An engine in idling and a man is speaking and then"
ref_cap = "A machine makes stitching sounds while people are talking in the background"

score, error_prob, penalized_score = evaluator.sentence_score(eval_cap, [ref_cap], return_error_prob=True)

print("Cand:", eval_cap)
print("Ref:", ref_cap)
print(f"SBERT sim: {score:.4f}, Error Prob: {error_prob:.4f}, Penalized score: {penalized_score:.4f}")

eval_cap = "An engine in idling and a man is speaking"
ref_cap = "A machine makes stitching sounds while people are talking in the background"

score, error_prob, penalized_score = evaluator.sentence_score(eval_cap, [ref_cap], return_error_prob=True)

print("Cand:", eval_cap)
print("Ref:", ref_cap)
print(f"SBERT sim: {score:.4f}, Error Prob: {error_prob:.4f}, Penalized score: {penalized_score:.4f}")

print("----Using larger models----")
evaluator = Evaluator(device='cpu', sbert_model='paraphrase-TinyBERT-L6-v2', echecker_model='echecker_clotho_audiocaps_base')

eval_cap = "An engine in idling and a man is speaking and then"
ref_cap = "A machine makes stitching sounds while people are talking in the background"

score, error_prob, penalized_score = evaluator.sentence_score(eval_cap, [ref_cap], return_error_prob=True)

print("Cand:", eval_cap)
print("Ref:", ref_cap)
print(f"SBERT sim: {score:.4f}, Error Prob: {error_prob:.4f}, Penalized score: {penalized_score:.4f}")

eval_cap = "An engine in idling and a man is speaking"
ref_cap = "A machine makes stitching sounds while people are talking in the background"

score, error_prob, penalized_score = evaluator.sentence_score(eval_cap, [ref_cap], return_error_prob=True)

print("Cand:", eval_cap)
print("Ref:", ref_cap)
print(f"SBERT sim: {score:.4f}, Error Prob: {error_prob:.4f}, Penalized score: {penalized_score:.4f}")