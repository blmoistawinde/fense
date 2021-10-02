import streamlit as st
from fense.evaluator import Evaluator
"""
# Audio Caption Evaluation

"""

if 'model' not in st.session_state:
    st.session_state['model'] = Evaluator(device='cpu', sbert_model='paraphrase-MiniLM-L6-v2', echecker_model='echecker_clotho_audiocaps_tiny')

ref_cap = st.text_input("Reference Caption:", "A machine makes stitching sounds while people are talking in the background")

eval_cap = st.text_input("Caption to Evaluate:", "An engine in idling and a man is speaking and then")

score, error_prob, penalized_score = st.session_state['model'].sentence_score(eval_cap, [ref_cap], return_error_prob=True)

col1, col2 = st.columns(2)
col1.metric("SBERT similarity", f"{score:.3f}")
if error_prob > 0.9:
    col2.metric("Error Probability", f"{error_prob:.3f}", "Penalize", delta_color='inverse')
else:
    col2.metric("Error Probability", f"{error_prob:.3f}")

st.write(f"Overall score: {penalized_score:.3f}")
my_bar = st.progress(penalized_score)