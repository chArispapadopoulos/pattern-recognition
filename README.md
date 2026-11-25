# Pattern Recognition & Machine Learning – Team 13

Course project, Aristotle University of Thessaloniki (2025-26). Team project.

## Parts A–C: estimation and classification from scratch (`Pattern Recognition Parts A-C - Code.ipynb`)
Implemented with NumPy only, without library estimators.

- **A – Maximum likelihood:** Gaussian parameters (μ, Σ) of three 2-D classes, 3D density plot and contour comparison. Discussion of the overlap (Bayes error) and why Mahalanobis distance fits better than Euclidean here.
- **B – Parzen windows:** hypercube and Gaussian kernels; h chosen by minimising the MSE against the true N(1, 4) density. Best h = 2.8 (hypercube, MSE 0.00179) and h = 0.8 (Gaussian, MSE 0.00111); the U-shaped MSE curves illustrate the bias–variance trade-off.
- **C – k-NN:** own implementation of distance, neighbour search and majority vote; k swept from 1 to 30, best k = 11; plots of the k-NN volume and of the non-linear decision boundaries.

## Part D: 5-class problem on a 224-feature dataset (`Pattern Recognition Part D - Code.ipynb`)
8,743 training samples, 80/20 hold-out split, pipelines with StandardScaler inside the folds, coarse-to-fine grid search with 5-fold CV, GPU training with cuML.

| Model | CV accuracy | Hold-out accuracy |
|---|---|---|
| MLP (256, 128), ReLU | 82.24% | 85.0% |
| Random Forest (500 trees, sqrt features) | 81.71% | – |
| XGBoost (1000 trees, depth 10) | – | 85.93% |
| SVM, RBF (C = 2.72, γ = 0.0058) | 85.07% | 86.74% |
| SVM, polynomial (degree 3, C = 6, coef0 = 0.3) | 85.74% | 88.34% |

The final predictor is a **majority-voting ensemble** of three polynomial SVMs (a bagged high-variance one, a plain baseline and a bootstrap one). The choice between models also weighed the number of support vectors, not only accuracy. Classes 2 and 5 remained the hardest to separate in every model.

Predictions for the unlabeled test set are in `labels13.npy`.

## Files
```
Pattern Recognition Parts A-C - Code.ipynb  parts A, B, C
Pattern Recognition Part D - Code.ipynb    part D
labels13.npy      test-set predictions
Project_Presentation.pdf   presentation slides (English translation)
```

## How to run
```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn xgboost
```
Part D was run on Colab with RAPIDS cuML for the GPU SVMs. The datasets are provided by the course and are not included in this repository. Place dataset1.csv, dataset2.csv, dataset3.csv, datasetTV.csv and datasetTest.csv next to the notebooks.

## Tech
Python, NumPy, scikit-learn, XGBoost, cuML, pandas, Matplotlib, Seaborn.
