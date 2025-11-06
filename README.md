# Machine Learning Classification Models Tutorial

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Latest-blue.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/sekertutku/Machine-Learning---Classifications-Models)

## Overview

This repository provides a comprehensive tutorial on machine learning classification techniques using Python. The project demonstrates 6 essential classification algorithms through a real-world medical diagnosis example (Breast Cancer Wisconsin Dataset) with detailed mathematical explanations, interactive visualizations using Plotly, and comprehensive evaluation metrics. Each model is explained with practical examples to showcase their unique strengths, use cases, and performance characteristics.

## 🎮 Interactive Demo

**👉 [Run the Interactive Notebook on Kaggle](https://www.kaggle.com/code/dandrandandran2093/machine-learning-classifications-models)**

*For the best experience with interactive Plotly visualizations and pre-configured datasets, use the Kaggle notebook above. All models are ready to run with visual explanations and evaluation metrics!*

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Classification Models](#classification-models)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [Mathematical Foundations](#mathematical-foundations)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [References](#references)

## Introduction

Classification is a supervised learning task that assigns data points to predefined categories or classes. This tutorial covers both classical statistical approaches and modern machine learning techniques for binary classification, providing a solid foundation for real-world classification problems, particularly in medical diagnosis where accuracy and interpretability are crucial.

## Dataset

### Breast Cancer Wisconsin (Diagnostic) Dataset

The tutorial utilizes the Breast Cancer Wisconsin dataset for all classification models:

- **Total Samples**: 569 tumors
- **Classes**: 
  - Benign (B) → 0 (357 samples)
  - Malignant (M) → 1 (212 samples)
- **Features**: 30 numerical features describing tumor characteristics
- **Purpose**: Binary classification for cancer diagnosis

**Key Features:**
- `radius_mean`: Mean of distances from center to perimeter points
- `texture_mean`: Standard deviation of gray-scale values
- `perimeter_mean`: Tumor perimeter measurement
- `area_mean`: Tumor area measurement
- `smoothness_mean`: Local variation in radius lengths
- `compactness_mean`: Perimeter² / area - 1.0
- `concavity_mean`: Severity of concave portions
- `concave_points_mean`: Number of concave portions
- And 22 more features...

**Why This Dataset?**
- Real-world medical importance
- Balanced classes (roughly 60-40 split)
- Multiple features for comparison
- Well-documented and widely used for benchmarking

## Classification Models

The tutorial covers 6 comprehensive classification techniques:

### 1. Logistic Regression
- **Concept**: Probabilistic classifier using sigmoid function
- **Formula**: `P(y=1|x) = 1 / (1 + e^(-z))` where `z = w^T x + b`
- **Use Case**: Binary classification with probability estimates
- **Implementation**: Both manual (from scratch) and sklearn versions

**Key Components:**
- **Sigmoid Function**: Converts linear output to probability (0-1)
- **Forward Propagation**: Computes predictions
- **Backward Propagation**: Updates weights using gradients
- **Cost Function**: Binary cross-entropy loss
- **Learning Rate**: 0.1
- **Iterations**: 50,000

**Advantages:**
- Interpretable coefficients
- Outputs probability scores
- Fast training and prediction

**Performance:** 98.25% accuracy

### 2. K-Nearest Neighbors (KNN)
- **Concept**: Instance-based learning using proximity
- **Formula**: `d = √(Σ(x₂-x₁)²)` (Euclidean distance)
- **Use Case**: Non-parametric classification
- **Hyperparameter Tuning**: Systematic K-value optimization

**Algorithm Steps:**
1. Choose K value (number of neighbors)
2. Calculate distance to all training points
3. Find K nearest neighbors
4. Count votes from each class
5. Assign majority class to test point

**Distance Metrics:**
- **Euclidean**: `√((x₂-x₁)² + (y₂-y₁)²)`
- Manhattan: `|x₂-x₁| + |y₂-y₁|`
- Minkowski: Generalized distance

**Key Insights:**
- **Optimal K = 9** (found through testing K=1 to K=14)
- Odd K values prevent tie situations
- Interactive visualization shows train vs test accuracy
- Lower K → Overfitting, Higher K → Underfitting

**Advantages:**
- No training phase (lazy learning)
- Simple and intuitive
- Works with non-linear boundaries

**Disadvantages:**
- Slow prediction for large datasets
- Sensitive to feature scaling
- Curse of dimensionality

**Performance:** 97.66% accuracy (at K=9)

### 3. Support Vector Machine (SVM)
- **Concept**: Finds optimal hyperplane with maximum margin
- **Use Case**: High-dimensional classification with clear margins
- **Kernel**: RBF (Radial Basis Function) for non-linear boundaries

**Key Concepts:**
- **Hyperplane**: Decision boundary separating classes
- **Support Vectors**: Critical points closest to boundary
- **Margin**: Distance between hyperplane and support vectors
- **Maximization**: SVM maximizes this margin for robustness

**How It Works:**
1. Find support vectors (nearest points to boundary)
2. Calculate margin between classes
3. Maximize margin to find optimal hyperplane
4. Classify new data based on which side of hyperplane

**Advantages:**
- Excellent with high-dimensional data (30 features)
- Memory efficient (uses only support vectors)
- Effective with clear margins

**Disadvantages:**
- Slower with large datasets
- Sensitive to noise and outliers
- Less interpretable than linear models

**Performance:** 98.83% accuracy (second-best model)

### 4. Naive Bayes Classification
- **Concept**: Probabilistic classifier based on Bayes' Theorem
- **Formula**: `P(Class|X) = [P(X|Class) × P(Class)] / P(X)`
- **Type**: Gaussian Naive Bayes (for continuous features)
- **Use Case**: Fast baseline with probability estimates

**Bayes' Theorem Components:**
- **P(Class|X)**: Posterior probability (what we want)
- **P(X|Class)**: Likelihood (features given class)
- **P(Class)**: Prior probability (class distribution)
- **P(X)**: Evidence (feature probability)

**"Naive" Assumption:**
- Assumes all features are independent
- Not usually true, but works surprisingly well
- Simplifies computation significantly

**Algorithm Steps:**
1. Calculate prior probability for each class
2. Calculate likelihood of features for each class
3. Apply Bayes' theorem
4. Choose class with highest posterior probability

**Types of Naive Bayes:**
- **Gaussian**: For continuous data (used here)
- Multinomial: For count data (text classification)
- Bernoulli: For binary features

**Advantages:**
- Extremely fast training and prediction
- Works well with small datasets
- Good for text classification
- Handles high-dimensional data

**Disadvantages:**
- Independence assumption rarely true
- Sensitive to zero probabilities
- Lower accuracy compared to ensemble methods

**Performance:** 93.57% accuracy (lowest but still excellent)

### 5. Decision Tree Classification
- **Concept**: Tree-like model with yes/no questions
- **Methodology**: CART (Classification and Regression Trees)
- **Use Case**: Interpretable decision rules
- **Splitting Criterion**: Gini impurity or Entropy

**How It Works:**
1. Start with all data at root
2. Find best feature to split (lowest Gini/Entropy)
3. Divide data into groups based on threshold
4. Repeat recursively on each group
5. Stop when pure or criteria met
6. Assign class label to each leaf

**Decision Process Example:**
```
         x2 > 15?
        /        \
      NO         YES
      /            \
  x1>25?         x1>25?
  /    \         /    \
Benign Malignant Malignant Benign
```

**Splitting Criteria:**
- **Gini Impurity**: Measures node purity
- **Entropy**: Information gain measurement
- **Variance**: For regression tasks

**Advantages:**
- Easy to understand and visualize
- No data normalization required
- Handles non-linear patterns naturally
- Shows feature importance

**Disadvantages:**
- Prone to overfitting (96.51% vs RF 98.84%)
- Unstable (small changes → different tree)
- Less accurate than ensemble methods

**Performance:** 96.51% accuracy

### 6. Random Forest Classification
- **Concept**: Ensemble of multiple decision trees
- **Methodology**: Bagging (Bootstrap Aggregating)
- **Parameters**: n_estimators=100, random_state=42
- **Use Case**: Production-ready robust predictions

**Bootstrap Aggregating Process:**
1. Create random subsets of data (with replacement)
2. Build decision tree for each subset
3. Use random feature selection at each split
4. Each tree makes independent prediction
5. Combine predictions by majority voting

**Formula:** 
```
Final Prediction = (1/n) × Σ Treeᵢ(x)  [Majority Vote]
```

**Key Features:**
- **n_estimators**: 100 trees in the forest
- **Bootstrap**: Random sampling with replacement
- **Feature Importance**: Identifies most predictive features
- **Aggregation**: Voting reduces variance

**Feature Importance Analysis:**
Top 5 most predictive features:
1. **area_worst** (15.8%): Largest tumor area
2. **perimeter_worst** (9.8%): Maximum perimeter
3. **concave points_worst** (9.7%): Worst boundary irregularity
4. **concave points_mean** (9.6%): Average irregularity
5. **perimeter_mean** (7.6%): Average perimeter

**Insight**: "Worst" measurements (extreme values) are stronger cancer indicators than mean values!

**Advantages:**
- **Best accuracy** (98.84%)
- Reduces overfitting vs single tree
- Provides feature importance scores
- Robust to outliers and noise
- Works for both classification & regression

**Disadvantages:**
- Slower prediction than single tree
- Less interpretable (black box)
- Requires more memory
- Longer training time

**Performance:** 98.84% accuracy (🏆 **Best Model**)

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
jupyter>=1.0.0
```

## Installation

### Option 1: Use Kaggle (Recommended) ⭐

The easiest way to explore this tutorial is on Kaggle where everything is pre-configured:

👉 **[Open Interactive Notebook on Kaggle](https://www.kaggle.com/code/dandrandandran2093/machine-learning-classifications-models)**

### Option 2: Run Locally

1. Clone the repository:
```bash
git clone https://github.com/sekertutku/Machine-Learning---Classifications-Models.git
cd Machine-Learning---Classifications-Models
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. **Dataset:**
   
   ✨ The Breast Cancer Wisconsin dataset is included in the `input/` directory:
   ```
   input/
   └── data.csv
   ```
   
   **Note:** The code uses Kaggle-specific paths (`/kaggle/input/...`). To run locally, update the dataset path in the code from:
   ```python
   df = pd.read_csv("/kaggle/input/tumors-dataset/data.csv")
   ```
   to:
   ```python
   df = pd.read_csv("input/data.csv")
   ```

## Usage

### On Kaggle (Recommended) ⭐

Simply open the [Kaggle notebook](https://www.kaggle.com/code/dandrandandran2093/machine-learning-classifications-models) and run the cells. All dependencies and datasets are pre-configured with interactive visualizations!

### Locally

#### Running the Complete Tutorial

Execute the main script to run all classification models:

```bash
python machine-learning-classifications-models.py
```

#### Running in Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook machine-learning-classifications-models.ipynb
```

### Making Predictions

Example usage for each model:

```python
# Logistic Regression (Manual Implementation)
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate=0.1, num_iterations=50000):
    # Initialize, train, and predict
    # Returns train and test accuracy
    pass

# Logistic Regression (Sklearn)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
prediction = log_reg.predict(X_test)

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

# Support Vector Machine
from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(X_train, y_train)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature Importance (Random Forest)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
```

## Key Features

### Comprehensive Model Comparison
- **6 Different Algorithms**: From simple to complex
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Visual error analysis
- **Feature Importance**: Top predictive features identified
- **Hyperparameter Tuning**: KNN K-value optimization

### Interactive Visualizations
- **Plotly Integration**: Interactive hover, zoom, and pan capabilities
- **Confusion Matrix Heatmap**: Seaborn visualization
- **KNN Accuracy Curves**: Train vs test accuracy for different K values
- **Feature Importance Bar Chart**: Top 10 features visualization
- **Scatter Plots**: Class distribution analysis

### Manual Implementation
- **Logistic Regression from Scratch**:
  - Forward and backward propagation
  - Gradient descent optimization
  - Cost function plotting
  - Comparison with sklearn implementation

### Mathematical Explanations
- **LaTeX Formulas**: Clear mathematical notation
- **Step-by-Step Algorithms**: Understanding the theory
- **Visual Examples**: Graphs demonstrating concepts
- **Probability Calculations**: Bayes' theorem breakdown

### Medical Context
- **Real-World Application**: Breast cancer diagnosis
- **Class Balance**: Stratified train-test split
- **Critical Metrics**: Focus on Recall (minimize missed cancers)
- **False Negative Analysis**: Only 1 cancer missed out of 32

### Code Quality
- **Clean Structure**: Well-organized and commented code
- **Reusable Components**: Modular design
- **Data Preprocessing**: Normalization and encoding
- **Best Practices**: Following scikit-learn conventions

## Model Performance

### Performance Summary

| Rank | Model | Accuracy | Key Strength |
|------|-------|----------|--------------|
| 🥇 1 | **Random Forest** | **98.84%** | Best overall, feature importance, robust |
| 🥈 2 | **SVM** | **98.83%** | High-dimensional performance, clear margins |
| 🥉 3 | **Logistic Regression** | **98.25%** | Fast, interpretable, probability scores |
| 4 | **KNN (K=9)** | **97.66%** | No training, simple, non-linear boundaries |
| 5 | **Decision Tree** | **96.51%** | Interpretable, visual rules, no scaling needed |
| 6 | **Naive Bayes** | **93.57%** | Extremely fast, good baseline |

**Performance Gap:** 5.27% difference between best (RF) and worst (NB) demonstrates importance of model selection!

### Confusion Matrix Analysis (Random Forest)

```
                Predicted
           Benign  Malignant
Actual
Benign        54        0      ← Perfect! No false alarms
Malignant      1       31      ← Only 1 cancer missed
```

**Key Metrics:**
- ✅ **True Negatives: 54** - All benign correctly identified
- ✅ **False Positives: 0** - No unnecessary patient anxiety
- ⚠️ **False Negatives: 1** - 1 malignant missed (97% Recall)
- ✅ **True Positives: 31** - 31 of 32 cancers caught

**Classification Report:**
```
              precision    recall  f1-score   support

      Benign       0.98      1.00      0.99        54
   Malignant       1.00      0.97      0.98        32

    accuracy                           0.99        86
```

**Medical Significance:**
- **Recall for Malignant: 97%** - Critical for cancer detection
- **Precision for Malignant: 100%** - Every cancer prediction was correct
- **Zero False Positives** - No healthy patients incorrectly diagnosed

### When to Use Each Model

**Random Forest** 🏆 Best Choice:
- Production deployment
- Maximum accuracy required (98.84%)
- Need feature importance analysis
- Robust to outliers
- **Use when:** Medical diagnosis, critical applications

**SVM** - High-Performance Alternative:
- Nearly identical accuracy (98.83%)
- Excellent with high-dimensional data (30 features)
- Memory efficient
- **Use when:** Large feature spaces, clear class separation

**Logistic Regression** - Clinical Baseline:
- Fast and interpretable (98.25%)
- Probability outputs for risk assessment
- Coefficients show feature impact
- **Use when:** Need explainability, quick deployment

**KNN** - Quick Prototyping:
- No training phase (97.66%)
- Easy to implement
- Works with non-linear boundaries
- **Use when:** Fast prototyping, small datasets

**Decision Tree** - Educational Tool:
- Easy to visualize (96.51%)
- Transparent decision rules
- But overfits compared to Random Forest
- **Use when:** Need interpretable rules, teaching

**Naive Bayes** - Speed Priority:
- Fastest prediction (93.57%)
- Works with limited data
- Good baseline
- **Use when:** Real-time systems, resource constraints

### Feature Importance Insights

**Top 10 Most Predictive Features (Random Forest):**

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | area_worst | 15.8% | Largest tumor area strongest predictor |
| 2 | perimeter_worst | 9.8% | Maximum boundary size |
| 3 | concave points_worst | 9.7% | Worst boundary irregularity |
| 4 | concave points_mean | 9.6% | Average irregularity |
| 5 | perimeter_mean | 7.6% | Average boundary |
| 6 | concavity_mean | 7.0% | Average concavity |
| 7 | radius_worst | 6.8% | Maximum radius |
| 8 | area_mean | 5.6% | Average area |
| 9 | radius_mean | 4.7% | Average radius |
| 10 | concavity_worst | 3.2% | Worst concavity |

**Key Insight:** "Worst" measurements (extreme values) account for ~45% of prediction power, indicating that extreme tumor characteristics are more indicative of malignancy than average measurements!

## Mathematical Foundations

### Logistic Regression

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```
Where: `z = w^T x + b`

**Cost Function (Binary Cross-Entropy):**
```
J(w,b) = -(1/m) × Σ[y log(ŷ) + (1-y) log(1-ŷ)]
```

**Gradient Descent Update:**
```
w := w - α × ∂J/∂w
b := b - α × ∂J/∂b
```

### K-Nearest Neighbors

**Euclidean Distance:**
```
d(p,q) = √(Σ(pᵢ - qᵢ)²)
```

**2D Distance:**
```
d = √((x₂-x₁)² + (y₂-y₁)²)
```

**3D Distance:**
```
d = √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)
```

### Naive Bayes

**Bayes' Theorem:**
```
P(Class|X) = [P(X|Class) × P(Class)] / P(X)
```

**Components:**
- **P(Class|X)**: Posterior probability (what we want)
- **P(X|Class)**: Likelihood (features given class)
- **P(Class)**: Prior probability
- **P(X)**: Evidence

### Random Forest

**Final Prediction (Classification):**
```
ŷ = mode{Tree₁(x), Tree₂(x), ..., Treeₙ(x)}
```

**Averaging (Regression):**
```
ŷ = (1/n) × Σ Treeᵢ(x)
```

## Evaluation Metrics

### Confusion Matrix

```
                Predicted Class
           Negative    Positive
Actual
Negative      TN         FP      ← False Positive (Type I Error)
Positive      FN         TP      ← False Negative (Type II Error)
```

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
**Interpretation:** Overall correctness percentage

**Example:** (31 + 54) / 86 = 98.84%

### Precision
```
Precision = TP / (TP + FP)
```
**Interpretation:** Of predicted positives, how many are actually positive?

**Medical Context:** Precision = 100% means no false alarms (critical for patient anxiety)

### Recall (Sensitivity, True Positive Rate)
```
Recall = TP / (TP + FN)
```
**Interpretation:** Of actual positives, how many did we catch?

**Medical Context:** Recall = 97% means we caught 31 of 32 cancers (most critical metric!)

### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
**Interpretation:** Harmonic mean balancing Precision and Recall

**Use Case:** Ideal when you need balance, especially with imbalanced classes

### Specificity (True Negative Rate)
```
Specificity = TN / (TN + FP)
```
**Interpretation:** Of actual negatives, how many did we correctly identify?

**Example:** 54 / 54 = 100% (perfect specificity)

### When to Prioritize Each Metric

**Accuracy:**
- ✅ Use when: Classes are balanced
- ❌ Avoid when: Imbalanced datasets (can be misleading)

**Precision:**
- ✅ Use when: False Positives are costly
- Example: Email spam (don't want to mark important emails as spam)

**Recall:**
- ✅ Use when: False Negatives are costly (our case!)
- Example: Cancer detection (don't want to miss sick patients)

**F1-Score:**
- ✅ Use when: Need balance between Precision and Recall
- ✅ Use when: Working with imbalanced data

## Key Insights

### Model Selection Strategy
1. **Start Simple**: Logistic Regression as interpretable baseline (98.25%)
2. **Try Alternatives**: Test KNN, SVM, Naive Bayes
3. **Ensemble Methods**: Random Forest for production (98.84%)
4. **Evaluate Thoroughly**: Use confusion matrix, not just accuracy

### Critical Findings

**🏆 Random Forest Wins Because:**
- Highest accuracy (98.84%)
- Only 1 false negative (97% Recall)
- Zero false positives (100% Precision for malignant)
- Provides feature importance (interpretability)
- Robust to overfitting

**📊 Feature Insights:**
- "Worst" measurements > mean values
- Top 5 features explain ~45% of predictions
- Tumor size and irregularity are key indicators

**⚠️ The One Miss:**
- 1 malignant tumor classified as benign
- Highlights need for:
  - Multiple diagnostic methods
  - Regular follow-ups
  - Lower decision thresholds for high-risk patients
  - Ensemble voting of multiple models

### Best Practices

**Data Preprocessing:**
- ✅ Min-Max normalization for distance-based models (KNN, SVM)
- ✅ Tree-based models don't require scaling
- ✅ Stratified train-test split (maintains class balance)
- ✅ Handle missing values (none in this dataset)

**Model Training:**
- ✅ Use random_state for reproducibility
- ✅ Hyperparameter tuning (KNN K-value)
- ✅ Compare manual vs library implementations (Logistic Regression)
- ✅ Cross-validation for robust evaluation

**Evaluation:**
- ✅ Don't rely on accuracy alone
- ✅ Analyze confusion matrix
- ✅ Focus on domain-critical metrics (Recall for cancer)
- ✅ Compare multiple models

**Production Deployment:**
- ✅ Use Random Forest for best accuracy
- ✅ Have SVM as backup/validation
- ✅ Monitor false negatives closely
- ✅ Consider ensemble voting
- ✅ Always combine with clinical judgment

### Common Pitfalls

❌ **Using Accuracy Alone**
- 95% accuracy can mean missing all cancer cases if 95% benign!

❌ **Ignoring Class Imbalance**
- Always check class distribution
- Use stratified split

❌ **Not Scaling Features**
- Critical for KNN, SVM, Logistic Regression
- Forget for Decision Trees, Random Forest

❌ **Overfitting Single Decision Tree**
- Random Forest reduces this (96.51% → 98.84%)

❌ **Ignoring Domain Context**
- In medicine, False Negatives > False Positives

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss proposed modifications.

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- Add more classification algorithms (XGBoost, LightGBM)
- Implement cross-validation
- Add GridSearchCV for hyperparameter tuning
- Create more interactive visualizations
- Add model deployment examples
- Improve documentation

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## References

### Course
- **Udemy**: MACHINE LEARNING by DATAI TEAM

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Scikit-learn Classification Guide](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Algorithms & Theory
- [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#classification)
- [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html#classification)
- [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Decision Trees](https://scikit-learn.org/stable/modules/tree.html#classification)
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

### Dataset
- [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

### Related Project
- 📈 **Machine Learning Regression Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-regression-models) [[GitHub]](https://github.com/sekertutku/Machine-Learning---Regression-Models)

## Acknowledgments

Special thanks to:
- DATAI TEAM for the comprehensive machine learning course
- UCI Machine Learning Repository for the dataset
- Scikit-learn developers for excellent ML library
- Plotly and Seaborn teams for visualization tools
- The open-source community for making these tools accessible
- Medical researchers who made this dataset available

---

**Note**: This tutorial is intended for educational purposes. The models and results should not be used for actual medical diagnosis without proper validation and clinical oversight. Always consult qualified healthcare professionals for medical decisions.

## 📞 Connect

If you have questions or suggestions:
- Open an issue in this repository
- Connect on [Kaggle](https://www.kaggle.com/dandrandandran2093)
- Visit my website: [tutkufurkan.com](https://www.tutkufurkan.com/)
- Star this repository if you found it helpful!

---

**Happy Learning! 🎯✨**

🌐 More projects at [tutkufurkan.com](https://www.tutkufurkan.com/)
