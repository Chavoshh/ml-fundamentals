# ml-fundamentals

Classical machine learning workflow on the California Housing dataset,
built as Week 2 of an 8-week ML learning plan. This project walks the
full pipeline from raw data to a tuned model with honest evaluation.

## What's in this repo

A single notebook — `notebooks/01_workflow.ipynb` — that covers:

- Data loading, exploration, and diagnostic plots
- Train/test split with strict discipline (test set touched exactly once)
- Baseline linear regression with RMSE interpreted in real-world units
- Batch gradient descent implemented from scratch in NumPy, verified
  against scikit-learn's closed-form solver
- Learning-rate experiments showing convergence and divergence over
  ~184 orders of magnitude
- Preprocessing with StandardScaler and Pipeline (fit-on-train discipline)
- Decision trees demonstrating overfitting empirically
- Random forest reducing variance through ensembling
- 5-fold cross-validation for model comparison
- GridSearchCV for hyperparameter tuning
- Final single-shot test-set evaluation

## Results

Best model: Random Forest, 200 estimators, max_depth=20, tuned via 5-fold
cross-validated grid search on the training set.

| Model                     | CV RMSE   | Test RMSE |
|---------------------------|-----------|-----------|
| Linear regression         | ~0.7250   | 0.7456    |
| Decision tree (depth=8)   | ~0.6500   | 0.6497    |
| Random Forest (tuned)     | 0.5103    | 0.5046    |

Test RMSE 0.5046 corresponds to a typical prediction error of ~$50k on
median California house values — a ~32% improvement over the linear
baseline.

The close agreement between CV RMSE (0.5103) and test RMSE (0.5046)
validates the experimental pipeline: no data leakage, no distribution
shift, honest model selection.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python and
dependency management.

```bash
git clone https://github.com/Chavoshh/ml-fundamentals.git
cd ml-fundamentals
uv sync
```

Open `notebooks/01_workflow.ipynb` in VS Code and select the `.venv`
kernel.

## What I learned

- The classical ML workflow end to end, including the discipline of
  train/validation/test separation
- Hand-coded gradient descent connects directly to iterative
  least-squares from estimation theory; Levenberg-Marquardt is the same
  idea with adaptive damping between gradient descent and Gauss-Newton
- The bias-variance tradeoff seen empirically across four model families
  (linear, unconstrained tree, depth-limited tree, random forest)
- Cross-validation gives more trustworthy performance estimates than
  single splits, and Pipelines prevent leakage during CV
- Hyperparameter tuning is usually the smallest of ML wins; data,
  problem formulation, and model choice matter more

## Environment

Python 3.12 (managed by uv), pandas, NumPy, scikit-learn, matplotlib.