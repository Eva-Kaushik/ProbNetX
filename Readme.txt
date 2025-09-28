# ProbNetX: Research-Grade Bayesian Network Framework

**Primary Author:** Rohit Kaushik, Data Analyst, Hanson Professional Services, USA  
**Secondary Author:** Eva Kaushik, Doctorate, University of Tennessee, Knoxville, USA

## Overview
I developed **ProbNetX**, a fully parameterizable Bayesian network framework that **learns probabilistic dependencies** from discrete datasets and performs **exact inference** to predict unobserved or missing variables. ProbNetX ensures **reproducibility, interpretability, and robustness**, making it suitable for high-dimensional or sparse datasets in both experimental and applied research.

## 1. Conditional Probability Table (CPT) Estimation (Learning from Data)

* **Inputs:** Network structure file (`.str`) and training CSV dataset.  
* **Methodology:** The algorithm parses the network’s DAG and computes each node’s **conditional probability distribution conditioned on its parent nodes**. **Laplace smoothing / pseudo-counts** ensure numerical stability and mitigate sparse-data artifacts.  
* **Outputs:** Fully specified CPTs for every node, exported in canonical topological order (`output.txt`).  

## 2. Exact Inference and Prediction

* **Inputs:** Network structure, learned CPTs, and test dataset with missing values.  
* **Inference:** Missing values are predicted using **Markov blanket exact inference**, computing posterior probabilities for each variable in a **probabilistically consistent and rigorous** manner.  
* **Outputs:** Completed test dataset (`completedTest.csv`) with imputed values, preserving instance order for reproducibility.

## 3. Theoretical Foundation

* **Bayesian Networks:** ProbNetX operates on discrete DAGs, representing conditional dependencies among variables. Each node \(X_i\) has a conditional probability table (CPT) \(P(X_i \mid Parents(X_i))\).  

* **CPT Learning:**  
For a node \(X_i\) with parents \(Parents(X_i)\), the conditional probabilities are estimated as:

\[
P(X_i=1 \mid Parents(X_i)) = \frac{Count(X_i=1, Parents) + \alpha}{Count(Parents) + 2\alpha}
\]

where \(\alpha\) is a pseudo-count (Laplace smoothing) to prevent zero-frequency issues.

* **Exact Inference:**  
For a query node \(Q\) with missing value, the **Markov blanket** \(MB(Q)\) is used to compute the exact posterior:

\[
P(Q=1 \mid MB(Q)) = \frac{P(Q=1) \prod_{C \in Children(Q)} P(C \mid Parents(C))}{\sum_{v=0}^{1} P(Q=v) \prod_{C \in Children(Q)} P(C \mid Parents(C))}
\]

This guarantees **exact posterior probabilities** without approximations.

## 4. Algorithm Overview (Pseudocode)

Input: Network structure S, training data D_train, test data D_test
Output: Learned CPTs, predicted values for missing entries
Parse DAG from structure S to create nodes N
For each node n in N:
a. Identify parent nodes P(n)
b. For each configuration of P(n):
i. Count occurrences in D_train
ii. Apply Laplace smoothing / pseudo-counts
iii. Compute conditional probabilities and store in CPT
For each instance x in D_test:
a. For each missing value node n_missing:
i. Compute posterior P(n_missing=1 | observed values) using Markov blanket
ii. Assign value 1 if P>=0.5 else 0
Save CPTs to output.txt
Save completed test data to completedTest.csv

## 5. Example: Naive Bayes Spam Classifier for ProbNetX

| Node   | Parents | CPT (smoothed)          |
| ------ | ------- | ---------------------- |
| Spam   | -       | [0.4, 0.6]             |
| Free   | Spam    | [0.8, 0.2], [0.1, 0.9] |
| Credit | Spam    | [0.7, 0.3], [0.2, 0.8] |

* If `Spam` is missing for a test instance, ProbNetX computes the **posterior probability** using observed features (`Free`, `Credit`) via **Markov blanket exact inference**.  
* Assign `Spam=1` if posterior ≥ 0.5; else assign `Spam=0`.

## 6. Key Advantages

1. **Fully Developed from Scratch:** Complete control over CPT computation and inference logic.  
2. **Exact Inference:** Guarantees probabilistic rigor without approximations.  
3. **Robust to Sparse Data:** Pseudo-counts prevent zero-frequency errors.  
4. **Reproducible and Research-Grade:** Outputs CPTs and predictions in canonical formats.  
5. **Generalizable:** Works with any discrete Bayesian network, from simple Naive Bayes to complex DAGs.  

## 7. Scientific Impact

ProbNetX provides a **reproducible, interpretable, and rigorous framework** for predictive modeling, decision-support, and hypothesis testing. It is suitable for **high-dimensional, sparse, or incomplete datasets** in domains such as bioinformatics, NLP, finance, or experimental research.