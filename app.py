"""
Web app for the Academic Question Difficulty Predictor.

Run locally:  streamlit run app.py
"""
import matplotlib
matplotlib.use("Agg")  # draw charts without opening windows

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Question Difficulty Predictor", page_icon="🧠", layout="centered")


@st.cache_resource
def load_model():
    # Importing the module trains the ANN once; Streamlit caches it after that.
    from src import question_difficulty_predictor as qdp
    return qdp


qdp = load_model()

st.title("🧠 Academic Question Difficulty Predictor")
st.caption("Neural Networks and Deep Learning (PCCCS582) · ANN (MLP) on TF-IDF features")

# ---------------- Predict ----------------
st.subheader("Try it")
examples = {
    "Pick an example...": "",
    "Easy": "What is a linked list?",
    "Moderate": "Explain the difference between an array and a linked list with examples.",
    "Difficult": "Derive the average-case time complexity of quicksort and prove it formally.",
}
choice = st.selectbox("Example questions", list(examples))
question = st.text_area("Type an academic question", value=examples[choice], height=100)

if st.button("Predict difficulty", type="primary"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        label, probs = qdp.predict(question.strip())
        colour = {"Easy": "green", "Moderate": "orange", "Difficult": "red"}[label]
        st.markdown(f"### Predicted difficulty: :{colour}[{label}]")
        chart = pd.DataFrame({"Probability": [probs[c] for c in qdp.CLASSES]}, index=qdp.CLASSES)
        st.bar_chart(chart)

# ---------------- Model details ----------------
st.divider()
st.subheader("How the model performs")

acc = accuracy_score(qdp.y_test, qdp.pred)
c1, c2, c3 = st.columns(3)
c1.metric("Test accuracy", f"{acc:.0%}")
c2.metric("Training questions", len(qdp.y_train))
c3.metric("Test questions", len(qdp.y_test))

tab1, tab2 = st.tabs(["Confusion matrix", "Training loss"])
with tab1:
    st.image("results/confusion_matrix.png")
with tab2:
    st.image("results/loss_curve.png")

with st.expander("How it works"):
    st.markdown(
        """
1. **TF-IDF** converts each question into numbers based on how important each word is.
2. **Extra features** are added: question length, and counts of keywords such as
   *derive / prove / analyze* (hard) and *explain / describe / compare* (moderate).
3. An **MLP neural network** with two hidden layers (64 and 32 neurons, ReLU, Adam optimizer)
   learns to map these features to *Easy*, *Moderate* or *Difficult*.
4. The model is evaluated on 25% of the questions that it never saw during training.
        """
    )