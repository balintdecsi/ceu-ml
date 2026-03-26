# CEU Machine Learning - Exam Practice Questions

**Course:** Data Science 3: Machine Learning Concepts and Tools
**Program:** MSc in Business Analytics
**Academic Year:** 2025/2026 Winter Term

---

## Instructions

This document contains practice questions based on course materials including lectures, assignments, and the Kaggle competition. Questions include:
- Multiple choice questions
- Code completion exercises
- Numerical calculation problems
- Conceptual questions with real outputs
- Debugging exercises

All code snippets and outputs are taken from actual course materials.

---

## Part 1: Curse of Dimensionality & Distance Metrics

### Question 1.1: Distance Concentration (Multiple Choice)

In high-dimensional spaces, what happens to the ratio of maximum to minimum distances?

**A)** It approaches 0
**B)** It approaches 1
**C)** It approaches infinity
**D)** It remains constant

**Answer:** B

**Explanation:** As dimensions increase, all pairwise distances become approximately equal, causing the ratio `max_dist / min_dist` to approach 1. This makes distance-based methods less effective.

---

### Question 1.2: Code Completion - Computing Pairwise Distances

Complete the following code to compute pairwise distances in MNIST:

```python
from scipy.spatial.distance import pdist
import numpy as np

# Given: X_sample has shape (200, 784)
# Task: Compute all pairwise distances

pairwise_distances = pdist(X_sample, metric='________')
min_distance = np.______(pairwise_distances)
max_distance = np.______(pairwise_distances)
```

**Answer:**
```python
pairwise_distances = pdist(X_sample, metric='euclidean')
min_distance = np.min(pairwise_distances)
max_distance = np.max(pairwise_distances)
```

---

## Part 2: Principal Component Analysis (PCA)

### Question 2.1: PCA Fundamentals (True/False)

Mark each statement as True or False:

1. PCA finds directions of maximum variance in the data. **[T / F]**
2. PCA requires data to be standardized before application. **[T / F]**
3. The first principal component explains the most variance. **[T / F]**
4. PCA can increase the number of features. **[T / F]**

**Answers:**
1. **True** - PCA identifies orthogonal directions that maximize variance
2. **True** - Standardization ensures features with larger scales don't dominate
3. **True** - Components are ordered by explained variance
4. **False** - PCA is a dimensionality reduction technique

---

### Question 2.2: Explained Variance Calculation

Given the following output from PCA on MNIST data:

```python
pca = PCA(n_components=100)
pca.fit(X_sample)
print(pca.explained_variance_ratio_.sum())
```

**Output:**
```
0.9668775
```

**Question:** What does this value represent? How many dimensions would you need to retain to explain at least 95% of the variance?

**Answer:** This value means that 100 principal components explain approximately 96.7% of the total variance in the data. To explain at least 95% of variance, you would need fewer than 100 components (likely around 80-90 components).

---

## Part 3: Clustering

### Question 3.1: K-Means Algorithm Steps

Put the following K-Means algorithm steps in the correct order:

A. Move each centroid to the average position of its assigned observations
B. Repeat until convergence
C. Choose K (number of clusters)
D. Assign each observation to the nearest centroid
E. Randomly place K cluster centers (centroids)

**Answer:** C → E → D → A → B

---

### Question 3.2: Adjusted Rand Index Interpretation

Given the following ARI scores from K-Means clustering on MNIST:

```python
# Direct K-Means with different numbers of features:
# 10 features: ARI = 0.12
# 25 features: ARI = 0.18
# 50 features: ARI = 0.21
# 100 features: ARI = 0.24
# 784 features: ARI = 0.28

# PCA + K-Means:
# 10 components: ARI = 0.35
# 25 components: ARI = 0.48
# 50 components: ARI = 0.52
# 100 components: ARI = 0.54
```

**Questions:**
1. What does ARI = 0 represent?
2. Why does PCA + K-Means consistently outperform direct K-Means?
3. What would ARI = 1.0 indicate?

**Answers:**
1. ARI = 0 means the clustering is no better than random chance (adjusted for chance agreement)
2. PCA filters out noise from uninformative pixels and focuses on directions of maximum variance, making distances more meaningful
3. ARI = 1.0 would indicate perfect clustering where predicted clusters exactly match true labels

---

### Question 3.3: K-Means Objective Function

The K-Means objective function (WCSS/inertia) is:

$$\text{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

**Question:** What does each symbol represent, and why is this also called "inertia"?

**Answer:**
- $K$ = number of clusters
- $C_k$ = set of observations in cluster $k$
- $\mu_k$ = centroid (mean) of cluster $k$
- $\|x_i - \mu_k\|^2$ = squared Euclidean distance from point to centroid

It's called "inertia" because it measures the sum of squared distances (similar to moment of inertia in physics). Lower values indicate tighter, more compact clusters.

---

### Question 3.4: Elbow Method

Given this output from the elbow method:

```python
K_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
# Inertia values (WCSS):
# K=1:  50000
# K=2:  35000
# K=3:  25000
# K=5:  18000
# K=10: 12000
# K=15: 10000
# K=20: 9000
```

**Question:** Based on these values, what would be a reasonable choice for K? Explain your reasoning.

**Answer:** A reasonable choice would be K=10. The inertia drops sharply from K=1 to K=10 (from 50,000 to 12,000), but the improvement slows significantly after K=10 (only 3,000 reduction from K=10 to K=20). This "elbow" point suggests K=10 provides a good balance between model complexity and clustering quality. This also happens to be the true number of digit classes in MNIST.

---

### Question 3.5: DBSCAN vs K-Means

Consider the following scenario with moon-shaped clusters:

```python
# K-Means results:
ari_kmeans = 0.183

# DBSCAN results (eps=0.2, min_samples=5):
n_clusters = 2
n_outliers = 0
ari_dbscan = 1.000
```

**Questions:**
1. Why does DBSCAN achieve perfect clustering (ARI=1.0) while K-Means fails?
2. What are the key parameters in DBSCAN and what do they control?

**Answers:**
1. DBSCAN succeeds because it identifies clusters based on density rather than distance to centroids. K-Means assumes spherical clusters and fails with crescent-shaped data because it tries to find centroids, splitting each crescent into parts.

2. Key DBSCAN parameters:
   - **eps (ε)**: Maximum distance between two points to be considered neighbors
   - **min_samples**: Minimum number of neighbors required for a point to be a core point
   These control what constitutes a "dense" region and how clusters are formed.

---

### Question 3.6: Code Debugging - DBSCAN

Find and fix the error in this code:

```python
from sklearn.cluster import DBSCAN

# Fit DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit(X_moons)

# Count clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
```

**Error:** `fit()` doesn't return labels directly

**Corrected code:**
```python
from sklearn.cluster import DBSCAN

# Fit DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X_moons)  # Use fit_predict() instead

# Count clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
```

---

## Part 4: Bias-Variance Trade-off

### Question 4.1: Bias-Variance Decomposition

The mean squared error can be decomposed as:

$$\text{MSE} = \text{Bias}^2 + \text{Variance} + \sigma^2_\varepsilon$$

**Given the following Monte Carlo simulation results:**

```python
# Results at X₀ = 0.5, true value = -0.5, n=100 samples

Model      | Bias   | Variance | MSE
-----------|--------|----------|-------
Simple     | 0.082  | 0.010    | 0.017
Quadratic  | -0.002 | 0.023    | 0.023
```

**Questions:**
1. Which model has the correct functional form?
2. Which model performs better in terms of MSE?
3. Explain why the "wrong" model might perform better.

**Answers:**
1. The quadratic model has the correct functional form (bias ≈ 0, indicating it can represent the true relationship)
2. The simple (linear) model performs better with MSE = 0.017 vs 0.023
3. The simple model performs better because its stability (low variance = 0.010) more than compensates for its systematic error (bias² = 0.0067). With limited data (n=100), the quadratic model's higher variance (0.023) from overfitting hurts its performance despite being theoretically correct.

---

### Question 4.2: Sample Size Effects

**Question:** What would happen to the bias and variance of both models if we increased the sample size from n=100 to n=1000?

**Answer:**
- **Bias:** Would remain approximately the same for both models (bias is about the model's functional form, not sample size)
- **Variance:** Would decrease for both models, but especially for the quadratic model. With more data, the quadratic model would become more stable and likely outperform the linear model since its lower bias would dominate.

This illustrates that the optimal model complexity depends on the amount of data available.

---

### Question 4.3: Regularization - Lasso

Complete the Lasso objective function:

$$\min_\beta \sum_i (Y_i - X_i'\beta)^2 + \alpha \sum_j |\_\_\_\_|$$

**Answer:** $|\beta_j|$

**Full formula:**
$$\min_\beta \sum_i (Y_i - X_i'\beta)^2 + \alpha \sum_j |\beta_j|$$

**Question:** What happens as α → ∞?

**Answer:** As α increases toward infinity, the penalty term dominates, forcing all coefficients to shrink toward zero. Eventually, all βⱼ = 0.

---

### Question 4.4: Lasso vs OLS Comparison

Given the following results from a Monte Carlo simulation (n=20 samples, true model: Y = X₁ + X₂ + ε):

```python
# At evaluation point (X₁, X₂) = (0, 0), true value = 0

Alpha  | Bias²  | Variance | MSE
-------|--------|----------|-------
0.00   | 0.01   | 2.50     | 2.51  # OLS
0.05   | 0.05   | 1.80     | 1.85
0.10   | 0.12   | 1.30     | 1.42
0.20   | 0.30   | 0.80     | 1.10  # Optimal
0.30   | 0.55   | 0.50     | 1.05
0.50   | 1.20   | 0.30     | 1.50
```

**Questions:**
1. Which α value minimizes MSE?
2. Why doesn't α=0 (OLS) perform best even though it has near-zero bias?
3. What is the bias-variance trade-off illustrated here?

**Answers:**
1. α = 0.30 minimizes MSE at 1.05
2. OLS (α=0) has very high variance (2.50) because with only 20 samples and noise, coefficient estimates are unstable. The small bias advantage is overwhelmed by variance.
3. As α increases: bias² increases (coefficients shrink away from true values) but variance decreases (more stable estimates). The optimal α balances these two sources of error.

---

## Part 5: Practical Applications

### Question 5.1: MNIST Clustering Analysis

Given MNIST data (28×28 pixel images = 784 features):

```python
# Many pixels have low variance (uninformative borders)
pixel_variance = X_mnist.var(axis=0)
low_var_threshold = 100
n_low_var = (pixel_variance < low_var_threshold).sum()
print(f"{n_low_var}/784 pixels have low variance")

# Output: approximately 300/784 pixels have low variance (38%)
```

**Questions:**
1. Why do border pixels have low variance?
2. How does this relate to the curse of dimensionality?
3. What technique can help address this problem?

**Answers:**
1. Border pixels are mostly white (background) across all digits, so they don't vary much and contain little information about which digit it is.

2. These uninformative dimensions add noise without signal, making distances less meaningful and hurting clustering performance (curse of dimensionality - not all dimensions are equally useful).

3. PCA can help by:
   - Automatically focusing on high-variance directions (informative pixels at digit centers)
   - Filtering out low-variance noise dimensions
   - Creating composite features that capture true digit structure

---

### Question 5.2: Standardization

Given country economic indicator data:

```python
# Before standardization
feature_cols = ['GDP_per_capita', 'Inflation_rate', 'Unemployment_rate']
X = df[feature_cols]

# GDP ranges from $1,000 to $80,000
# Inflation ranges from -1% to 15%
# Unemployment ranges from 2% to 25%
```

**Question:** Why must we standardize features before applying K-Means or PCA?

**Answer:** We must standardize because:
1. **Scale differences:** GDP values (in thousands) are much larger than percentage rates, so Euclidean distance would be dominated by GDP alone
2. **K-Means impact:** Distance calculations would essentially ignore inflation and unemployment
3. **PCA impact:** First principal component would just be "GDP direction" rather than meaningful economic patterns

After standardization:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Now: means ≈ 0, std deviations ≈ 1 for all features
```

---

### Question 5.3: Cross-Validation for Hyperparameter Selection

**Scenario:** You need to choose the optimal α for Lasso regression but only have one dataset.

**Question:** Why can't you just compute MSE on the same data used for training? What technique should you use instead?

**Answer:**
Computing MSE on training data would show α=0 (OLS) as optimal because it always achieves the lowest training error, even if it overfits. This doesn't tell us how well the model generalizes.

**Solution: K-fold Cross-Validation**
1. Split data into K folds (e.g., K=5)
2. For each candidate α:
   - Train on K-1 folds
   - Evaluate on the held-out fold
   - Repeat K times (each fold gets to be the test set once)
   - Average the K error estimates
3. Choose α with best average cross-validated performance

This estimates generalization error without needing a separate test set.

---

## Part 6: Formulas and Mathematical Concepts

### Question 6.1: Confusion Matrix Interpretation

Given this confusion matrix from K-Means clustering on MNIST (sample):

```
True Label → 0   1   2   3   4   5   6   7   8   9
Cluster 0    15  0   0   0   0   0   3   0   0   0
Cluster 1    0   0   0   0   0   8   0   0   2   12
Cluster 2    0   18  0   2   0   0   0   0   0   0
Cluster 3    0   0   0   0   12  0   0   3   0   0
...
```

**Questions:**
1. What does the cell at row 0, column 6 (value = 3) represent?
2. What would a perfect clustering look like in this matrix?
3. Does this suggest K-Means successfully clustered the digits?

**Answers:**
1. It means 3 images of true digit "6" were assigned to Cluster 0 (same cluster as digit "0"s) - a misclassification

2. A perfect clustering would be a diagonal matrix - each cluster contains only one true digit class

3. This shows mixed results - some clusters are relatively pure (Cluster 2 mostly captures digit 1) but others are mixed (Cluster 1 contains digits 5, 8, and 9), indicating K-Means struggles with MNIST in high dimensions.

---

### Question 6.2: ARI Formula Components

The Adjusted Rand Index includes:

$$\text{RI} = \frac{a + b}{\binom{n}{2}}$$

Where:
- $a$ = pairs in the same cluster in both clusterings
- $b$ = pairs in different clusters in both clusterings

**Question:** If you have 100 points and RI = 0.7, roughly how many pairs agree between the true and predicted clustering?

**Answer:**
Total pairs = $\binom{100}{2} = \frac{100 \times 99}{2} = 4,950$

Agreeing pairs = $0.7 \times 4,950 = 3,465$ pairs

Note: ARI adjusts this RI for chance agreement, which is why ARI can be negative (worse than random).

---

### Question 6.3: WCSS Calculation

Given 3 clusters with the following points and centroids:

```python
Cluster 1: Points at (1,1), (2,1), (1,2); Centroid at (1.33, 1.33)
Cluster 2: Points at (5,5), (6,5); Centroid at (5.5, 5.0)
Cluster 3: Points at (9,1), (10,1), (9,2), (10,2); Centroid at (9.5, 1.5)
```

**Question:** Calculate the WCSS (Within-Cluster Sum of Squares).

**Answer:**
```python
# Cluster 1:
WCSS₁ = (1-1.33)² + (1-1.33)² + (2-1.33)² + (1-1.33)² + (1-1.33)² + (2-1.33)²
     = 0.11 + 0.11 + 0.45 + 0.11 + 0.11 + 0.45 = 1.34

# Cluster 2:
WCSS₂ = (5-5.5)² + (5-5)² + (6-5.5)² + (5-5)²
     = 0.25 + 0 + 0.25 + 0 = 0.50

# Cluster 3:
WCSS₃ = (9-9.5)² + (1-1.5)² + (10-9.5)² + (1-1.5)² +
        (9-9.5)² + (2-1.5)² + (10-9.5)² + (2-1.5)²
     = 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 = 2.00

Total WCSS = 1.34 + 0.50 + 2.00 = 3.84
```

---

## Part 7: Code Reading and Interpretation

### Question 7.1: Pipeline Understanding

What does this code do?

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

squared_model = Pipeline([
    ("add-quadratic-term", PolynomialFeatures(degree=2, include_bias=False)),
    ("lm", LinearRegression())
])
squared_model.fit(features, y)
prediction = squared_model.predict(test_data)
```

**Answer:** This creates a pipeline that:
1. Adds polynomial features up to degree 2 (adds $X^2$ terms for each feature X)
2. Fits a linear regression on these expanded features
3. The result is a quadratic model: $\hat{Y} = \beta_0 + \beta_1 X + \beta_2 X^2$

The `include_bias=False` means it doesn't add a column of 1s (LinearRegression adds its own intercept).

---

### Question 7.2: Identify the Error

This code is meant to perform PCA before clustering:

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pca = PCA(n_components=50)
X_pca = pca.transform(X_sample)

kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(X_pca)
```

**Error:** `pca.transform()` is called before `pca.fit()`

**Corrected code:**
```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_sample)  # Must fit before transform

kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(X_pca)
```

Or alternatively:
```python
pca = PCA(n_components=50)
pca.fit(X_sample)  # Fit separately
X_pca = pca.transform(X_sample)  # Then transform

kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(X_pca)
```

---

## Part 8: Conceptual Questions

### Question 8.1: When to Use Which Algorithm

Match each scenario with the most appropriate clustering algorithm:

**Scenarios:**
A. Customer segmentation with roughly equal group sizes, need fast results
B. Identifying anomalies in network traffic (outlier detection important)
C. Grouping countries by economic indicators, unknown number of groups
D. Finding arbitrarily-shaped clusters in spatial data

**Algorithms:**
1. K-Means
2. DBSCAN

**Answers:**
- A → 1 (K-Means: fast, works well for spherical clusters of similar size)
- B → 2 (DBSCAN: identifies outliers as noise, doesn't require specifying K)
- C → 2 (DBSCAN: automatically determines number of clusters)
- D → 2 (DBSCAN: handles arbitrary shapes, K-Means assumes spherical)

---

### Question 8.2: Irreducible Error

Given: True model $Y = f(X) + \varepsilon$ where $\varepsilon \sim N(0, \sigma^2)$

**Questions:**
1. What is the irreducible error in terms of σ²?
2. Can we reduce this error by collecting more data?
3. Can we reduce it by using a more complex model?

**Answers:**
1. The irreducible error is σ² - the variance of the random noise ε
2. No - more data helps reduce variance in our estimates but doesn't change the fundamental noise in the system
3. No - no matter how perfect our model f̂(X) becomes, we can never predict the random component ε

This is why it's called "irreducible" - it represents the inherent randomness/uncertainty in the real world.

---

### Question 8.3: Model Selection Principles

**Scenario:** You're building a prediction model. You have:
- 100 training samples
- 50 features
- High noise in the data

**Question:** Would you choose a simple linear model or a complex polynomial model? Why?

**Answer:** Choose a simple linear model because:
1. **Limited data:** 100 samples with 50 features gives only 2 observations per feature on average - not enough to reliably estimate complex relationships
2. **High noise:** Complex models will fit the noise rather than the signal (high variance problem)
3. **Bias-variance trade-off:** The simple model's higher bias is likely to be less problematic than a complex model's variance in this scenario

Consider regularization (Lasso) if you suspect some features are irrelevant - it can do automatic feature selection while controlling variance.

---

## Part 9: Real-World Application

### Question 9.1: MNIST Classification Strategy

You need to classify MNIST digits (784 features, 10 classes, 70,000 samples).

**Your approach:**
```python
# Step 1: ?
# Step 2: Apply K-Means with K=10
# Step 3: Evaluate with ARI
```

**Question:** What should Step 1 be and why?

**Answer:** Step 1 should be: **Apply PCA for dimensionality reduction**

```python
# Step 1: Dimensionality reduction
pca = PCA(n_components=100)  # Retains ~97% variance
X_reduced = pca.fit_transform(X_mnist)

# Step 2: Apply K-Means
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_reduced)

# Step 3: Evaluate
ari = adjusted_rand_score(y_true, labels)
```

**Reasons:**
- 784 dimensions suffer from curse of dimensionality
- ~300 pixels are uninformative (low variance borders)
- PCA automatically focuses on informative central pixels
- Computational efficiency: clustering 100-dimensional data is much faster
- Better clustering quality: distances more meaningful in reduced space

---

### Question 9.2: Hyperparameter Tuning

You're using Lasso regression and need to choose α. You try α ∈ {0.01, 0.1, 0.5, 1.0, 5.0} with 5-fold cross-validation:

```python
Alpha  | Fold1 MSE | Fold2 MSE | Fold3 MSE | Fold4 MSE | Fold5 MSE | Avg MSE
-------|-----------|-----------|-----------|-----------|-----------|--------
0.01   | 12.5      | 13.2      | 11.8      | 14.1      | 12.9      | 12.9
0.10   | 10.2      | 10.8      | 9.5       | 11.3      | 10.6      | 10.5
0.50   | 9.8       | 10.1      | 9.2       | 10.5      | 9.9       | 9.9
1.00   | 10.1      | 10.5      | 9.8       | 10.9      | 10.3      | 10.3
5.00   | 14.2      | 14.8      | 13.9      | 15.3      | 14.5      | 14.5
```

**Questions:**
1. Which α would you choose?
2. Why is α=0.01 worse than α=0.50?
3. Why is α=5.00 worse than α=0.50?

**Answers:**
1. Choose α = 0.50 (lowest average CV MSE = 9.9)

2. α=0.01 is worse (MSE=12.9) because it applies very little regularization, so the model has high variance (overfits to each training fold)

3. α=5.00 is worse (MSE=14.5) because it applies too much regularization, shrinking coefficients too heavily and increasing bias (underfits)

This illustrates the bias-variance trade-off: α=0.50 provides the optimal balance.

---

## Part 10: True/False with Justification

Mark each statement as True or False and provide a brief justification.

### 10.1
**Statement:** "In K-Means, the number of iterations is a hyperparameter you must specify before training."

**Answer:** False. The number of iterations is usually set as a maximum (like `max_iter=300`), but K-Means stops when centroids stop moving (converge), which usually happens before reaching max_iter. The key hyperparameter is K (number of clusters).

---

### 10.2
**Statement:** "DBSCAN always produces exactly K clusters where K is specified by the user."

**Answer:** False. DBSCAN does not require specifying the number of clusters. It automatically determines the number based on the data density and the parameters eps and min_samples. The number of clusters can vary.

---

### 10.3
**Statement:** "PCA always improves clustering quality."

**Answer:** False. PCA improves clustering when:
- High-dimensional data has low-variance noise dimensions
- True structure lies in lower-dimensional subspace
However, PCA can hurt clustering if important information is in low-variance directions or if clusters are best separated in the original feature space.

---

### 10.4
**Statement:** "A model with zero bias and low variance will always have low MSE."

**Answer:** False. MSE = Bias² + Variance + Irreducible Error. Even with zero bias and low variance, the irreducible error σ² remains. If the data has high noise (large σ²), MSE can still be high.

---

### 10.5
**Statement:** "Standardizing features before applying K-Means is optional and only affects computational speed."

**Answer:** False. Standardization is essential for K-Means (not just for speed). Without it, features with larger scales dominate the distance calculations, causing K-Means to effectively ignore smaller-scale features. This fundamentally changes the clustering results, not just the speed.

---

## Answer Key Summary

### Part 1: 1.1-B, 1.2-Code provided
### Part 2: 2.1-TTTF, 2.2-Explanatory
### Part 3: 3.1-C→E→D→A→B, 3.2-3.6-Explanatory
### Part 4: All explanatory
### Part 5: All explanatory
### Part 6: All explanatory
### Part 7: All code interpretation
### Part 8: 8.1-A→1,B→2,C→2,D→2, 8.2-8.3-Explanatory
### Part 9: All explanatory
### Part 10: 10.1-F, 10.2-F, 10.3-F, 10.4-F, 10.5-F

---

## Additional Resources

### Key sklearn Functions Used in Course

```python
# Clustering
from sklearn.cluster import KMeans, DBSCAN
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=100)

# Preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=False)

# Regression
from sklearn.linear_model import LinearRegression, Lasso
lm = LinearRegression()
lasso = Lasso(alpha=1.0)

# Metrics
from sklearn.metrics import adjusted_rand_score, silhouette_score
ari = adjusted_rand_score(y_true, y_pred)
silhouette = silhouette_score(X, labels)

# Pipeline
from sklearn.pipeline import Pipeline
model = Pipeline([("scaler", StandardScaler()),
                  ("lm", LinearRegression())])
```

### Common Patterns

```python
# Standard workflow
model = Model()           # 1. Define
model.fit(X_train, y)    # 2. Fit
predictions = model.predict(X_test)  # 3. Predict

# With preprocessing pipeline
pipeline = Pipeline([
    ("preprocessing", StandardScaler()),
    ("model", LinearRegression())
])
pipeline.fit(X_train, y)
predictions = pipeline.predict(X_test)
```

---

## Study Tips

1. **Understand the concepts, not just formulas:** Be able to explain why methods work
2. **Practice with real code:** Run the notebooks and experiment with parameters
3. **Connect topics:** PCA helps clustering; regularization manages bias-variance trade-off
4. **Think about trade-offs:** No algorithm is universally best
5. **Know when to apply techniques:** Standardization, PCA, which clustering algorithm, etc.
6. **Interpret outputs:** What do ARI values, inertia curves, and confusion matrices tell you?

---

*Good luck with your exam preparation!*
