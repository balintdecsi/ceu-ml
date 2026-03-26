# CEU Machine Learning - Quick Reference Cheat Sheet

**Course:** Data Science 3: Machine Learning Concepts and Tools

---

## Essential Formulas

### 1. Bias-Variance Decomposition
$$\text{MSE} = \text{Bias}^2 + \text{Variance} + \sigma^2_{\varepsilon}$$

- **Bias:** How far off is model on average?
- **Variance:** How much does model change with different data?
- **Irreducible error (σ²):** Random noise we cannot predict

### 2. K-Means Objective (WCSS/Inertia)
$$\text{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

Where:
- K = number of clusters
- C_k = observations in cluster k
- μ_k = centroid of cluster k

### 3. Adjusted Rand Index (ARI)
$$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$$

$$\text{RI} = \frac{a + b}{\binom{n}{2}}$$

- a = pairs in same cluster in both clusterings
- b = pairs in different clusters in both clusterings
- **ARI = 1:** Perfect clustering
- **ARI = 0:** No better than random
- **ARI < 0:** Worse than random

### 4. Lasso Objective Function
$$\min_\beta \sum_i (Y_i - X_i'\beta)^2 + \alpha \sum_j |\beta_j|$$

- α = penalty parameter (higher α = more shrinkage)
- α = 0 → OLS (no penalty)
- α → ∞ → all coefficients shrink to 0

---

## Key Algorithms

### K-Means Clustering

**Algorithm:**
1. Choose K (number of clusters)
2. Initialize: Random K centroids
3. **Repeat until convergence:**
   - **Assignment:** Assign each point to nearest centroid
   - **Update:** Move centroids to mean of assigned points

**Properties:**
- ✅ Fast, simple, scalable
- ✅ Works well for spherical clusters
- ❌ Must specify K in advance
- ❌ Sensitive to initialization
- ❌ Assumes spherical clusters
- ❌ All points assigned (no outlier detection)

**When to use:** Known K, roughly spherical/similar-sized clusters, need speed

### DBSCAN (Density-Based Clustering)

**Parameters:**
- **eps (ε):** Maximum distance between neighbors
- **min_samples:** Minimum points to form dense region

**Point Types:**
- **Core point:** Has ≥ min_samples neighbors within radius ε
- **Border point:** Within ε of core point (but not core itself)
- **Noise point:** Neither core nor border (labeled -1)

**Properties:**
- ✅ Finds arbitrary-shaped clusters
- ✅ Identifies outliers automatically
- ✅ No need to specify K
- ❌ Sensitive to parameter choice
- ❌ Struggles with varying densities
- ❌ Slower than K-Means

**When to use:** Unknown K, arbitrary shapes, need outlier detection

### PCA (Principal Component Analysis)

**Purpose:** Dimensionality reduction by finding directions of maximum variance

**Key Concepts:**
- First PC explains most variance
- PCs are orthogonal (uncorrelated)
- Filters out low-variance noise
- Makes distances more meaningful

**Properties:**
- ✅ Reduces dimensionality
- ✅ Removes multicollinearity
- ✅ Speeds up algorithms
- ❌ Components hard to interpret
- ❌ Linear combinations only
- ⚠️ Requires feature scaling

---

## Essential Code Patterns

### 1. Standard Workflow
```python
# Define → Fit → Predict/Transform
model = Model()
model.fit(X_train, y)
predictions = model.predict(X_test)
```

### 2. K-Means Clustering
```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Define and fit
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# Evaluate (if true labels available)
ari = adjusted_rand_score(y_true, labels)

# Access results
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_
```

### 3. DBSCAN
```python
from sklearn.cluster import DBSCAN

# Define and fit
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Count clusters and outliers
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = np.sum(labels == -1)
```

### 4. PCA
```python
from sklearn.decomposition import PCA

# Define and fit
pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X)

# Check explained variance
print(pca.explained_variance_ratio_.sum())  # Total variance explained

# Get components
components = pca.components_  # Shape: (n_components, n_features)
```

### 5. Standardization (CRITICAL!)
```python
from sklearn.preprocessing import StandardScaler

# Always standardize before K-Means, PCA, or distance-based methods
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verify: means ≈ 0, std ≈ 1
print(X_scaled.mean(axis=0))  # Should be near 0
print(X_scaled.std(axis=0))   # Should be near 1
```

### 6. Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Combine preprocessing and model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression())
])

# Fit pipeline
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### 7. Lasso Regression
```python
from sklearn.linear_model import Lasso

# Basic usage
lasso = Lasso(alpha=1.0, random_state=42)
lasso.fit(X_train, y_train)
predictions = lasso.predict(X_test)

# Check which coefficients are non-zero
n_nonzero = np.count_nonzero(lasso.coef_)
```

### 8. Elbow Method for Choosing K
```python
inertias = []
K_range = range(1, 21)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot and look for "elbow"
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.show()
```

### 9. Monte Carlo Simulation
```python
n_iterations = 1000
predictions = np.empty(n_iterations)

for i in range(n_iterations):
    # Generate new dataset
    X, y = generate_data(sample_size=100)

    # Fit model
    model.fit(X, y)

    # Predict at test point
    predictions[i] = model.predict(test_data)[0]

# Calculate bias and variance
bias = np.mean(predictions - true_value)
variance = np.var(predictions)
mse = np.mean((predictions - true_value)**2)
```

---

## Quick Decision Guide

### Should I standardize?
**YES** for: K-Means, DBSCAN, PCA, Lasso, distance-based methods
**NO** for: Decision trees, Random Forests (they're scale-invariant)

### K-Means or DBSCAN?
| Use K-Means if: | Use DBSCAN if: |
|-----------------|----------------|
| Know # clusters | Don't know # clusters |
| Spherical clusters | Arbitrary shapes |
| Similar cluster sizes | Need outlier detection |
| Need speed | Can sacrifice speed |

### Should I apply PCA before clustering?
**YES** if:
- High-dimensional data (p > 50)
- Many low-variance/noise features
- Computational constraints
- Distance concentration observed

**NO** if:
- Already low-dimensional (p < 10)
- All features informative
- Need interpretable features

### Simple or complex model?
| Prefer Simple Model | Prefer Complex Model |
|---------------------|---------------------|
| Small dataset (n < 100) | Large dataset (n > 1000) |
| High noise | Low noise |
| Few features | Many features |
| Need interpretability | Need accuracy |

### Which α for Lasso?
Use **cross-validation**:
1. Try α ∈ {0.001, 0.01, 0.1, 1.0, 10.0}
2. 5-fold or 10-fold CV
3. Choose α with lowest CV error
4. Lower α if too much shrinkage
5. Higher α if overfitting

---

## Common Mistakes to Avoid

❌ **Forgetting to standardize before K-Means/PCA**
→ Features with larger scales dominate

❌ **Using fit_transform() on test data**
→ Causes data leakage
→ Use fit() on train, transform() on test

❌ **Not checking for convergence in K-Means**
→ May need more iterations or different initialization

❌ **Confusing fit() and fit_predict()**
→ fit() returns model, fit_predict() returns labels

❌ **Using training error to select hyperparameters**
→ Always use cross-validation

❌ **Treating ARI like accuracy**
→ ARI adjusts for chance, can be negative

❌ **Forgetting random_state**
→ Results not reproducible

---

## Key Metrics & Evaluation

### Clustering Quality
```python
# If true labels available
from sklearn.metrics import adjusted_rand_score, silhouette_score

ari = adjusted_rand_score(y_true, labels)  # 1=perfect, 0=random
silhouette = silhouette_score(X, labels)   # -1 to 1, higher better

# Within-cluster sum of squares
inertia = kmeans.inertia_  # Lower is better
```

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)  # 1=perfect, 0=baseline, <0=very bad
```

---

## Important Constants & Defaults

### sklearn Default Parameters
```python
KMeans(n_clusters=8, n_init=10, max_iter=300, random_state=None)
DBSCAN(eps=0.5, min_samples=5)
PCA(n_components=None)  # None means keep all
Lasso(alpha=1.0, max_iter=1000)
StandardScaler()  # mean=0, std=1
```

### Rules of Thumb
- **PCA components:** Keep 90-95% explained variance
- **DBSCAN min_samples:** Start with 2 × dimensions + 1
- **K-Means n_init:** Use 10-50 (multiple random starts)
- **Sample size:** Need ~10 observations per feature minimum

---

## Curse of Dimensionality

**Problem:** In high dimensions:
1. Data becomes sparse (moves to edges of space)
2. Distances become meaningless (all approximately equal)
3. Volume concentrates in corners
4. Need exponentially more data

**Solutions:**
- Apply PCA to reduce dimensions
- Use feature selection
- Regularization (Lasso)
- Domain knowledge to remove irrelevant features

**Key Insight:** More features ≠ better model
→ Quality > Quantity

---

## Mathematical Notation Quick Reference

| Symbol | Meaning |
|--------|---------|
| $X_i$ | Feature vector for observation i |
| $Y_i$ | Target value for observation i |
| $\hat{Y}_i$ | Predicted value for observation i |
| $\beta$ | Coefficient vector |
| $\mu_k$ | Centroid of cluster k |
| $\varepsilon$ | Random error/noise |
| $\sigma^2$ | Variance |
| $\|\|x\|\|$ | Euclidean norm (length) of vector x |
| $\|\|x\|\|^2$ | Squared Euclidean distance |
| $\mathbb{E}[\cdot]$ | Expected value (average) |
| $\sum_i$ | Sum over all observations i |
| $\sum_j$ | Sum over all features j |
| $\binom{n}{2}$ | Combinations: n choose 2 = n(n-1)/2 |

---

## Debugging Checklist

When results look wrong, check:

1. ✅ Data standardized? (for K-Means/PCA/Lasso)
2. ✅ Correct fit/transform split? (fit on train only)
3. ✅ Random state set? (for reproducibility)
4. ✅ Correct metric? (ARI for clustering, MSE for regression)
5. ✅ No data leakage? (test data not seen during training)
6. ✅ Convergence reached? (check n_iter_ attribute)
7. ✅ Outliers handled? (may need DBSCAN or preprocessing)
8. ✅ Correct shape? (n_samples × n_features)

---

## Import Reference

```python
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Clustering
from sklearn.cluster import KMeans, DBSCAN

# Dimensionality reduction
from sklearn.decomposition import PCA

# Preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Linear models
from sklearn.linear_model import LinearRegression, Lasso

# Metrics
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    mean_squared_error,
    confusion_matrix
)

# Utilities
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import pdist
```

---

**Remember:** Understanding concepts > memorizing formulas!

*Good luck on your exam!*
