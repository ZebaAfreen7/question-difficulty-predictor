"""
Academic Question Difficulty Predictor (PCCCS582)
ANN (MLP) on TF-IDF text features -> Easy / Moderate / Difficult

Install:  pip install scikit-learn matplotlib numpy
Run:      python src/question_difficulty_predictor.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ---------------- 1. Dataset (replace/extend with your own questions) ----------------
EASY = [
    "Define an array.", "What is a variable?", "Name the four pillars of OOP.",
    "What does CPU stand for?", "Define a neural network.", "What is an operating system?",
    "State Ohm's law.", "What is a compiler?", "List any two types of memory.",
    "What is the full form of HTML?", "Define machine learning.", "What is a database?",
    "Name two sorting algorithms.", "What is an IP address?", "Define a function in programming.",
    "What is a neuron in a neural network?", "Write the formula for the area of a circle.",
    "What is binary number system?", "Define a stack.", "What is the use of a loop?",
]
MODERATE = [
    "Explain the difference between a stack and a queue with examples.",
    "Describe how the backpropagation algorithm updates weights.",
    "Compare supervised and unsupervised learning with suitable examples.",
    "Explain the working of binary search and state its time complexity.",
    "Differentiate between TCP and UDP protocols.",
    "Explain how normalization improves database design.",
    "Describe the role of activation functions in a neural network.",
    "Explain the concept of inheritance with a code example.",
    "Write a program to reverse a linked list and explain each step.",
    "Describe the process of page replacement in operating systems.",
    "Explain overfitting and discuss two techniques to reduce it.",
    "Describe the working of a convolutional layer in a CNN.",
    "Explain the difference between process and thread.",
    "Illustrate how a hash table handles collisions.",
    "Explain the architecture of a recurrent neural network.",
    "Discuss the advantages of using dropout during training.",
    "Describe how gradient descent minimizes a loss function.",
    "Explain the ACID properties of a transaction with examples.",
    "Write an algorithm for BFS and trace it on a small graph.",
    "Compare Sigmoid, Tanh and ReLU activation functions.",
]
DIFFICULT = [
    "Derive the backpropagation equations for a three-layer network using the chain rule and analyze the vanishing gradient problem.",
    "Prove that the halting problem is undecidable and discuss its implications for program verification.",
    "Analyze the time and space complexity of Dijkstra's algorithm using a Fibonacci heap and justify the amortized bounds.",
    "Design a deadlock-free resource allocation scheme for a distributed system and prove its correctness.",
    "Evaluate the trade-offs between LSTM and Transformer architectures for long-sequence modelling and justify your choice for a real-time system.",
    "Derive the evidence lower bound (ELBO) of a variational autoencoder and explain the reparameterization trick.",
    "Critically analyze the convergence properties of Adam versus SGD with momentum on non-convex loss surfaces.",
    "Formulate the generative adversarial game as a minimax problem and derive the optimal discriminator.",
    "Prove that the greedy algorithm yields an optimal solution for the fractional knapsack problem.",
    "Design and analyze a consensus protocol tolerant to Byzantine faults and compare it with Paxos.",
    "Derive the gradient of the cross-entropy loss with softmax output and show how it simplifies.",
    "Evaluate how batch normalization changes the loss landscape and justify with mathematical reasoning.",
    "Construct a proof that every NP-complete problem reduces to 3-SAT and discuss its consequences.",
    "Analyze the security of RSA against chosen-ciphertext attacks and propose a mitigation with justification.",
    "Synthesize a complete pipeline for detecting anomalies in streaming data and critically evaluate its scalability.",
    "Derive the time complexity of the Master theorem cases and apply it to a non-trivial recurrence.",
    "Compare and critically evaluate attention mechanisms in sequence models and derive the scaled dot-product formulation.",
    "Prove the correctness of the Bellman-Ford algorithm and explain how it detects negative cycles.",
    "Design an optimal indexing strategy for a large relational database and analyze its query performance trade-offs.",
    "Analyze the bias-variance decomposition of expected squared error and derive it formally.",
]

texts = EASY + MODERATE + DIFFICULT
labels = (["Easy"] * len(EASY) + ["Moderate"] * len(MODERATE) + ["Difficult"] * len(DIFFICULT))
CLASSES = ["Easy", "Moderate", "Difficult"]

# ---------------- 2. Extra hand-crafted features ----------------
HARD_WORDS = {"derive", "prove", "analyze", "analyse", "evaluate", "critically", "justify",
              "design", "formulate", "construct", "synthesize", "optimal"}
MID_WORDS = {"explain", "describe", "compare", "differentiate", "illustrate", "discuss", "write"}


def extra_features(qs):
    rows = []
    for q in qs:
        w = q.lower().replace(",", " ").replace(".", " ").replace("?", " ").split()
        rows.append([len(w) / 30.0,
                     sum(x in HARD_WORDS for x in w),
                     sum(x in MID_WORDS for x in w)])
    return np.array(rows)


# ---------------- 3. Train / test ----------------
X_train_t, X_test_t, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels)

vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
Xtr = np.hstack([vec.fit_transform(X_train_t).toarray(), extra_features(X_train_t)])
Xte = np.hstack([vec.transform(X_test_t).toarray(), extra_features(X_test_t)])

model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                      max_iter=800, random_state=42)
model.fit(Xtr, y_train)

pred = model.predict(Xte)
print(f"Test accuracy: {accuracy_score(y_test, pred):.2%}\n")
print(classification_report(y_test, pred, labels=CLASSES, zero_division=0))

# ---------------- 4. Plots (saved in results/) ----------------
os.makedirs("results", exist_ok=True)

ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=CLASSES),
                       display_labels=CLASSES).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")

plt.figure()
plt.plot(model.loss_curve_)
plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title("Training Loss Curve")
plt.savefig("results/loss_curve.png", dpi=150, bbox_inches="tight")
print("Saved results/confusion_matrix.png and results/loss_curve.png")


# ---------------- 5. Predict new questions ----------------
def predict(question):
    x = np.hstack([vec.transform([question]).toarray(), extra_features([question])])
    probs = model.predict_proba(x)[0]
    return str(model.classes_[probs.argmax()]), {str(c): round(float(v), 2) for c, v in zip(model.classes_, probs)}


if __name__ == "__main__":
    print("\nType a question to predict its difficulty (or 'q' to quit).")
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() == "q":
            break
        if not q:
            continue
        label, p = predict(q)
        print(f"Predicted: {label}   Probabilities: {p}")