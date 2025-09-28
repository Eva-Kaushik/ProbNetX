# ProbNetX: Research-Grade Bayesian Network Framework

Primary Author: Rohit Kaushik, Data Analyst, Hanson Professional Services, USA | Secondary Author: Eva Kaushik, PhD, University of Tennessee, Knoxville, USA

## Overview
ProbNetX is a fully parameterizable Bayesian network framework designed to learn probabilistic dependencies from discrete datasets and perform exact inference to predict missing or unobserved variables. It emphasizes reproducibility, interpretability, and robustness, making it ideal for high-dimensional or sparse datasets in both experimental and applied research.

### 1. Conditional Probability Table (CPT) Estimation
Inputs:
Network structure file (.str)
Training dataset (.csv)
Methodology: Parses the network’s DAG to identify parent-child relationships.
Computes each node’s conditional probability distribution conditioned on its parents.
Uses Laplace smoothing / pseudo-counts to ensure numerical stability and reduce sparse-data artifacts.
Outputs: Fully specified CPTs for all nodes, exported in canonical topological order (output.txt).

### 2. Exact Inference and Prediction
Inputs: Network structure
Learned CPTs
Test dataset with missing values
Inference: Missing values are predicted using Markov blanket exact inference, calculating posterior probabilities rigorously and probabilistically consistently.
Outputs: Completed test dataset (completedTest.csv) with imputed values, preserving instance order for reproducibility.
 
Guarantees exact posterior probabilities without approximations.

### 3. Algorithm Overview (Pseudocode)
Input: Network structure S, training data D_train, test data D_test
Output: Learned CPTs, predicted values for missing entries

1. Parse DAG from S to create nodes N
2. For each node n in N:
     a. Identify parent nodes P(n)
     b. For each configuration of P(n):
          i. Count occurrences in D_train
         ii. Apply Laplace smoothing / pseudo-counts
        iii. Compute and store conditional probabilities in CPT
3. For each instance x in D_test:
     a. For each missing node n_missing:
          i. Compute posterior P(n_missing=1 | observed values) via Markov blanket
         ii. Assign 1 if P >= 0.5 else 0
4. Save CPTs to output.txt
5. Save completed test data to completedTest.csv

### 4. Key Advantages
Exact Inference: Full control over CPT computation and inference logic and Probabilistically rigorous predictions without approximations.
Robust to Sparse Data: Pseudo-counts prevent zero-frequency errors.
Reproducible & Research-Grade: Outputs CPTs and predictions in canonical formats.
Generalizable: Works with any discrete Bayesian network—from simple Naive Bayes to complex DAGs.

Note: ProbNetX is a reproducible, interpretable, and rigorous framework for predictive modeling, decision-support, and hypothesis testing. It is suitable for high-dimensional, sparse, or incomplete datasets in domains such as: Bioinformatics, Natural Language Processing, Finance and Experimental research
