#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-06T22:37:06.755Z
"""

# # 🧠 INSTRUCTION
# 1. [Logistic Regression](#1)
# 2. [KNN Algorithm](#2)
# 3. [Support Vector Machine (SVM)](#3)
# 4. [Naive Bayes Classification](#4)
# 5. [Decision Tree](#5)
# 6. [Random Forest Classification](#6)
# 7. [Evaluation Classifications Model](#7)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"
import seaborn as sns
# machine learning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# <a id = "1"></a>
# # Logistic Regression


# Import data
tumors_data = pd.read_csv("/kaggle/input/tumors-dataset/data.csv")

tumors_data.head()

tumors_data.info()

# Drop unnecessary colons
tumors_data.drop(["Unnamed: 32","id"],axis = 1, inplace = True,errors='ignore')

# Classification
tumors_data["diagnosis"] = [1 if each == "M" else 0 for each in tumors_data.diagnosis]

tumors_data.info()

# Split data
y = tumors_data.diagnosis.values

x = tumors_data.drop(["diagnosis"],axis = 1)

print(y)

# Normalization
x_norm = (x - x.min()) / (x.max() - x.min())

# Train, test, split
x_train, x_test, y_train, y_test = train_test_split(x_norm,y,test_size = 0.2, random_state = 42)

print(y_test)

# Train, test, split
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

# Paramater initialize and sigmaid function
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b


# w,b = initialize_weights_and_bias(30)
# print(w)

def sigmoid(z):
    y_head = 1/(1 + np.exp(-z))
    return y_head


# print(sigmoid(0))
# print(sigmoid(6))

# In backward propagation we will use y_head that found in forward progation
# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation
# Forward and Backward Propagation
def forward_backward_propagation(w, b, x_train, y_train):
    # Forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = (np.sum(loss)) / x_train.shape[1]  # x_train.shape[1] is for scaling
    
    # Backward propagation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost, gradients

# Updating (learning) parameters
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    # Updating parameters for number_of_iteration times
    for i in range(number_of_iteration):
        # Make forward and backward propagation and find cost and gradients
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        
        # Update parameters
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        
        if i % 2500 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" % (i, cost))
    
    # Store updated parameters
    parameters = {"weight": w, "bias": b}
    
    # Plot cost
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters, gradients, cost_list

# Prediction
def predict(w, b, x_test):
    # Forward propagation
    z = sigmoid(np.dot(w.T, x_test) + b)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    
    # Convert probabilities to binary predictions
    # If z > 0.5, predict 1, else predict 0
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    
    return Y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    # Initialize parameters
    dimension = x_train.shape[0]
    w, b = initialize_weights_and_bias(dimension)
    
    # Train the model
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    # Make predictions
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)
    
    # Print train/test accuracy
    print("train accuracy: % {}".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: % {}".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

logistic_regression(x_train, y_train, x_test, y_test, learning_rate=0.1, num_iterations=50000)

# ============================
# Compare manual implementation vs sklearn library
# ============================
# Above: Manual implementation with forward/backward propagation
# Below: Using sklearn's optimized LogisticRegression
# Logistic Regression Model
logistic_reg = LogisticRegression()

logistic_reg.fit(x_train.T,y_train.T)

print("test accuracy: % {}".format(logistic_reg.score(x_train.T,y_train.T) * 100))

# <a id = "2"></a>
# # KNN (K-Nearest Neighbour)


# Import data
tumors_data_knn = pd.read_csv("/kaggle/input/tumors-dataset/data.csv")

tumors_data_knn.drop(["Unnamed: 32","id"],axis = 1, inplace = True,errors='ignore')
tumors_data_knn.tail()
# M = Malignant => Bad Tumor
# B = Benign => Good Tumor

# Split dataset into two classes
M = tumors_data_knn[tumors_data_knn["diagnosis"] == "M"]
B = tumors_data_knn[tumors_data_knn["diagnosis"] == "B"]

print(M.info())
print(B.info())

# Scatter Plot
plt.scatter(M.radius_mean,M.area_mean,color = "red",label="bad tumors")
plt.scatter(B.radius_mean,B.area_mean,color = "green",label="good tumors")
plt.xlabel("radius_mean")
plt.ylabel("area_mean")
plt.legend()

# Scatter Plot
plt.scatter(M.radius_mean,M.texture_mean,color = "red",label="bad tumors",alpha = 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color = "green",label="good tumors",alpha = 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()

# ### Algorithm Logic
# 
# KNN = K nearest neighbour
# 
# ### Steps:
# 
# 1) **Choose K value:** k = 3, k = 5, k = 4, etc.
# 2) **Find K nearest data points**
# 3) **Count how many points from each class among K nearest neighbors**
# 4) **Classify the test point based on majority class**
# 
# ---
# 
# ### Distance Calculation
# 
# ### Euclidean Distance
# 
# The straight-line distance between two points.
# 
# **Formula:**
# $$d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$$
# 
# **Example Calculation:**
# 
# Given points:
# - x1 = 142
# - x2 = 145
# - y1 = 0.12
# - y2 = 0.11
# 
# **For 3D space:**
# $$d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2 + (z_2-z_1)^2}$$
# 
# **Result:** 
# - Distance = 142
# - 9 + 0.0001 = 9.0001 ≈ 3
# 
# ---
# 
# ### Visualization Example
# ```python
# # KNN Classification Example
# # Green points: class = 3 (kotu)
# # Orange points: class = 5 (iyi)
# # Test point (shown in square) is classified using k=3
# # Looking at 3 nearest neighbors to determine majority class
# ```
# 
# **Important Note:** Choosing an odd K value prevents tie situations.
# 
# ---
# 
# ### KNN Parameters
# 
# - **n_neighbors:** K value (e.g., 3, 5, 7)
# - **metric:** Distance measurement method (euclidean, manhattan, minkowski)
# - **weights:** 'uniform' or 'distance' (gives more weight to closer neighbors)


# Split data
tumors_data_knn = tumors_data.copy()  # Create a new copy of the data
# tumors_data_knn.diagnosis = [1 if each == "M" else 0 for each in tumors_data.diagnosis]
y_knn = tumors_data_knn.diagnosis.values  # Target variable (already 0 and 1)
x_knn = tumors_data_knn.drop(["diagnosis"], axis=1)  # Feature variables

# Normalization
x_norm_knn = (x_knn - x_knn.min()) / (x_knn.max() - x_knn.min())  # Min-Max normalization

# Train, test, split - stratify maintains class balance
x_train_knn, x_test_knn, y_train_knn, y_test_knn = train_test_split(
    x_norm_knn, y_knn, test_size=0.3, random_state=42, stratify=y_knn
)

# Check class distribution
print("y_knn distribution:")
unique, counts = np.unique(y_knn, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count}")

# KNN Model
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train_knn,y_train_knn)

# Prediction
prediction = knn.predict(x_test_knn)
print("{} knn score: {}".format(3,knn.score(x_test_knn,y_test_knn)))

# Compare training and testing accuracy for different k values
train_scores = []
test_scores = []

for k in range(1, 15):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train_knn, y_train_knn)
    
    train_scores.append(knn_model.score(x_train_knn, y_train_knn))
    test_scores.append(knn_model.score(x_test_knn, y_test_knn))

# Create dataframe for plotting
df_comparison = pd.DataFrame({
    'K Values': list(range(1, 15)) * 2,
    'Accuracy': train_scores + test_scores,
    'Type': ['Train'] * 14 + ['Test'] * 14
})

# Plot with Plotly
fig = px.line(df_comparison, 
              x='K Values', 
              y='Accuracy',
              color='Type',
              markers=True,
              title='KNN: Training vs Testing Accuracy')

fig.update_traces(marker=dict(size=10),
                  line=dict(width=3))

fig.update_layout(
    xaxis_title="K Values",
    yaxis_title="Accuracy",
    hovermode='x unified',
    template='plotly_white',
    xaxis=dict(tickmode='linear', tick0=1, dtick=1),
    legend=dict(title="Dataset")
)

fig.show()

# Find best k value based on test accuracy
best_k = test_scores.index(max(test_scores)) + 1
print(f"\nBest k value: {best_k}")
print(f"Training Accuracy: % {train_scores[best_k-1]*100}")
print(f"Testing Accuracy: % {test_scores[best_k-1]*100}")

# <a id = "3"></a>
# # Support Vector Machine (SVM)


# ### What is SVM?
# 
# SVM finds the **best line/plane** that separates classes with **maximum margin**.
# 
# ### Key Concepts:
# - **Hyperplane:** Decision boundary separating classes
# - **Support Vectors:** Critical points closest to boundary (circled in image)
# - **Margin:** Distance between hyperplane and support vectors
# - **Max Margin Classifier:** SVM maximizes this margin
# 
# ---
# 
# ### How SVM Works
# 
# 1. Find support vectors (nearest points to boundary)
# 2. Calculate margin between classes
# 3. Maximize margin to find optimal hyperplane
# 4. Classify new data based on which side of hyperplane
# 
# ---
# 
# ### Pros & Cons
# 
# **Advantages:**
# - Works well in high dimensions
# - Effective with clear margins
# - Memory efficient
# 
# **Disadvantages:**
# - Slower with large datasets
# - Sensitive to noise


# Import data
tumors_data_svm = pd.read_csv("/kaggle/input/tumors-dataset/data.csv")
tumors_data_svm.drop(["Unnamed: 32","id"],axis = 1, inplace = True,errors='ignore')

# Split dataset into two classes
M = tumors_data_svm[tumors_data_svm["diagnosis"] == "M"]
B = tumors_data_svm[tumors_data_svm["diagnosis"] == "B"]

# Split data
tumors_data_svm = tumors_data.copy()  # Create a new copy of the data
# tumors_data_knn.diagnosis = [1 if each == "M" else 0 for each in tumors_data.diagnosis]
y_svm = tumors_data_svm.diagnosis.values  # Target variable (already 0 and 1)
x_svm = tumors_data_svm.drop(["diagnosis"], axis=1)  # Feature variables

# Normalization
x_norm_svm = (x_svm - x_svm.min()) / (x_svm.max() - x_svm.min())  # Min-Max normalization

# Train, test, split - stratify maintains class balance
x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(
    x_norm_svm, y_svm, test_size=0.3, random_state=42, stratify=y_svm
)

# Check class distribution
print("y_svm distribution:")
unique, counts = np.unique(y_svm, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count}")

# SVM Model
svm = SVC(random_state=1)
svm.fit(x_train_svm,y_train_svm)

# Model evaluation
print(f"accuracy of svm algorithm: % {svm.score(x_test_svm,y_test_svm)*100}")

# <a id = "4"></a>
# # Naive Bayes Classification


# ### What is Naive Bayes?
# 
# A **probabilistic classifier** that uses **Bayes' Theorem** to predict class probabilities.
# 
# **"Naive"** because it assumes all features are independent (not usually true, but works well anyway!)
# 
# ---
# 
# ### Bayes' Theorem
# 
# $$P(\text{Class}|X) = \frac{P(X|\text{Class}) \times P(\text{Class})}{P(X)}$$
# 
# **Simple terms:**
# - **P(Class|X):** What we want to find
# - **P(X|Class):** Likelihood of features given class
# - **P(Class):** Prior probability
# - **P(X):** Evidence
# 
# ---
# 
# ### How It Works
# 
# 1. Calculate prior probability for each class
# 2. Calculate likelihood of features for each class
# 3. Apply Bayes' theorem
# 4. Choose class with highest probability
# 
# ---
# 
# ### Types
# 
# - **Gaussian:** For continuous data (most common)
# - **Multinomial:** For count data (text/word frequency)
# - **Bernoulli:** For binary features (yes/no)
# 
# ---
# 
# ### Example
# 
# **Teacher Classification:**
# - P(Math Teacher) = 6/11 = 54%
# - P(Physics Teacher) = 5/11 = 46%
# 
# If person is tall:
# - P(Math|tall) = 75%
# - P(Physics|tall) = 25%
# 
# **Prediction:** Math Teacher ✅
# 
# ---
# 
# ### Pros & Cons
# 
# **Advantages:**
# - Fast and efficient
# - Works well with small datasets
# - Good for text classification
# 
# **Disadvantages:**
# - Assumes independence (unrealistic)
# - Sensitive to zero probabilities


# Import data
tumors_data_nbc = pd.read_csv("/kaggle/input/tumors-dataset/data.csv")
tumors_data_nbc.drop(["Unnamed: 32","id"],axis = 1, inplace = True,errors='ignore')

# Split dataset into two classes
M = tumors_data_nbc[tumors_data_nbc["diagnosis"] == "M"]
B = tumors_data_nbc[tumors_data_nbc["diagnosis"] == "B"]

# Split data
tumors_data_nbc = tumors_data.copy()  # Create a new copy of the data
# tumors_data_knn.diagnosis = [1 if each == "M" else 0 for each in tumors_data.diagnosis]
y_nbc = tumors_data_nbc.diagnosis.values  # Target variable (already 0 and 1)
x_nbc = tumors_data_nbc.drop(["diagnosis"], axis=1)  # Feature variables

# Normalization
x_norm_nbc = (x_nbc - x_nbc.min()) / (x_nbc.max() - x_nbc.min())  # Min-Max normalization

# Train, test, split - stratify maintains class balance
x_train_nbc, x_test_nbc, y_train_nbc, y_test_nbc = train_test_split(
    x_norm_nbc, y_nbc, test_size=0.3, random_state=42, stratify=y_nbc
)

# Check class distribution
print("y_nbc distribution:")
unique, counts = np.unique(y_nbc, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count}")

# Naive Bayes Model
nbc = GaussianNB()
nbc.fit(x_train_nbc,y_train_nbc)

# Model evaluation
print(f"accuracy of nbc algorithm: % {nbc.score(x_test_nbc,y_test_nbc)*100}")

# <a id = "5"></a>
# # Decision Tree


# ### What is Decision Tree?
# 
# A **tree-like model** that makes predictions by asking yes/no questions about features, splitting data at each step until reaching a final decision.
# 
# ---
# 
# ### CART (Classification and Regression Trees)
# 
# Decision trees have two types:
# 
# - **Classification:** Predicts categories (mor/mavi, benign/malignant)
# - **Regression:** Predicts continuous numbers (price, temperature)
# 
# ---
# 
# ### How It Works
# 
# **Step-by-step process:**
# 
# 1. Start with all data at the root
# 2. Find the best feature to split (e.g., x2 > 15?)
# 3. Divide data into two groups: "yes" and "no"
# 4. Repeat splitting on each group
# 5. Stop when groups are pure or criteria met
# 6. Assign class label to each leaf
# 
# **Visual Example:**
# ```
#          x2 > 15?
#         /        \
#       no         yes
#       /            \
#   x1>25?         x1>25?
#   /    \         /    \
# mor   mavi    mavi   mor
# ```
# 
# **In the image:** Data is split at x1=25 and x2=15, creating regions for mor (purple) and mavi (blue) classes.
# 
# ---
# 
# ### Pros & Cons
# 
# **Advantages:**
# - Easy to understand and visualize
# - No data scaling needed
# - Handles non-linear patterns
# 
# **Disadvantages:**
# - Prone to overfitting
# - Small data changes = big tree changes
# - Less accurate than ensemble methods


# Import data
tumors_data_dc = pd.read_csv("/kaggle/input/tumors-dataset/data.csv")
tumors_data_dc.drop(["Unnamed: 32","id"],axis = 1, inplace = True,errors='ignore')

# Classification
tumors_data_dc["diagnosis"] = [1 if each == "M" else 0 for each in tumors_data_dc.diagnosis]

# Split data
y_dc = tumors_data_dc.diagnosis.values
x_dc = tumors_data_dc.drop(["diagnosis"],axis = 1)

# Normalization
x_norm_dc = (x_dc - x_dc.min()) / (x_dc.max() - x_dc.min())  # Min-Max normalization

# Train, test, split - stratify maintains class balance
x_train_dc, x_test_dc, y_train_dc, y_test_dc = train_test_split(
    x_norm_dc, y_dc, test_size=0.15, random_state=42, stratify=y_dc
)

dc = DecisionTreeClassifier()
dc.fit(x_train_dc,y_train_dc)

# Model evaluation
print(f"accuracy of dc algorithm: % {dc.score(x_test_dc,y_test_dc)*100}")

# <a id = "6"></a>
# # Random Forest Classification


# ### What is Random Forest?
# An **ensemble learning method** that builds **multiple decision trees** and combines their predictions.
# **"Forest"** because it creates many trees and merges them together for more accurate and stable predictions!
# 
# ---
# 
# ### Bootstrap Aggregating (Bagging)
# $$\text{Final Prediction} = \frac{1}{n}\sum_{i=1}^{n} \text{Tree}_i\text{ (Majority Vote)}$$
# 
# **Simple terms:**
# - **Bootstrap:** Random sampling with replacement
# - **Aggregating:** Combining predictions from all trees
# - **Result:** More stable and accurate than single tree
# 
# ---
# 
# ### How It Works
# 1. Create random subsets of training data (with replacement)
# 2. Build a decision tree for each subset
# 3. Use random feature selection at each node split
# 4. Each tree makes independent prediction
# 5. Combine predictions by majority voting
# 
# ---
# 
# ### Key Parameters
# - **n_estimators:** Number of trees (default: 100)
# - **max_depth:** Maximum tree depth (prevent overfitting)
# - **max_features:** Features to consider per split
# - **min_samples_split:** Minimum samples to split node
# 
# ---
# 
# ### Example
# **Forest with 3 Trees:**
# - Tree 1 → x1>20: YES → x1>25: YES → Predicts: **mor** 
# - Tree 2 → x2>15: YES → x1>25: NO → Predicts: **mor**
# - Tree 3 → x2>15: NO → Predicts: **mavi**
# 
# **Final Prediction:** mor ✅ (2 votes vs 1)
# 
# ---
# 
# ### Pros & Cons
# **Advantages:**
# - Reduces overfitting (averaging multiple trees)
# - Handles large datasets efficiently
# - Provides feature importance scores
# - Works for both classification & regression
# 
# **Disadvantages:**
# - Slower prediction than single tree
# - Less interpretable (black box)
# - Requires more memory
# - May overfit on noisy data


# ============================
# Using same preprocessing from Decision Tree section above
# ============================

# Import data
tumors_data_dc = pd.read_csv("/kaggle/input/tumors-dataset/data.csv")
tumors_data_dc.drop(["Unnamed: 32","id"],axis = 1, inplace = True,errors='ignore')

# Classification
tumors_data_dc["diagnosis"] = [1 if each == "M" else 0 for each in tumors_data_dc.diagnosis]

# Split data
y_dc = tumors_data_dc.diagnosis.values
x_dc = tumors_data_dc.drop(["diagnosis"],axis = 1)

# Normalization
x_norm_dc = (x_dc - x_dc.min()) / (x_dc.max() - x_dc.min())  # Min-Max normalization

# Train, test, split - stratify maintains class balance
x_train_dc, x_test_dc, y_train_dc, y_test_dc = train_test_split(
    x_norm_dc, y_dc, test_size=0.15, random_state=42, stratify=y_dc
)

dc = DecisionTreeClassifier()
dc.fit(x_train_dc,y_train_dc)

# Model evaluation
print(f"accuracy of dc algorithm: % {dc.score(x_test_dc,y_test_dc)*100}")

# Random Forest
rf = RandomForestClassifier(n_estimators = 100,random_state=42)
rf.fit(x_train_dc,y_train_dc)

# Model evaluation
print(f"accuracy of rf algorithm: % {rf.score(x_test_dc,y_test_dc)*100}")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': x_dc.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

fig = px.bar(feature_importance, 
             x='Importance', 
             y='Feature',
             orientation='h',
             title='Top 10 Most Important Features (Random Forest)')
fig.show()

# <a id = "7"></a>
# # Evaluation Classifications Model


# ### What is Confusion Matrix?
# A **performance measurement tool** for classification models that shows how well predictions match actual values.
# It reveals **where and how often** the model makes mistakes, not just overall accuracy.
# 
# ---
# 
# ### The Four Outcomes
# 
# $$
# \begin{bmatrix}
# TN & FP \\
# FN & TP
# \end{bmatrix}
# $$
# 
# **True Negative (TN):** Correctly predicted benign (0) ✅
# 
# **False Positive (FP):** Benign wrongly predicted as malignant ❌
# 
# **False Negative (FN):** Malignant wrongly predicted as benign ❌ (Most dangerous!)
# 
# **True Positive (TP):** Correctly predicted malignant (1) ✅
# 
# ---
# 
# ### Core Metrics
# 
# **Accuracy:** Overall correctness
# $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
# 
# **Precision:** Of predicted malignant, how many are actually malignant?
# $$Precision = \frac{TP}{TP + FP}$$
# 
# **Recall (Sensitivity):** Of actual malignant, how many did we catch?
# $$Recall = \frac{TP}{TP + FN}$$
# 
# **F1-Score:** Balance between Precision and Recall
# $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$
# 
# ---
# 
# ### Understanding the Heatmap
# 
# - **Rows (y_true):** Actual diagnosis (0=Benign, 1=Malignant)
# - **Columns (y_pred):** Predicted diagnosis
# - **Diagonal:** Correct predictions (darker = better)
# - **Off-diagonal:** Errors (should be lighter)
# 
# ---
# 
# ### Why It Matters
# 
# **False Negative (FN) is critical in medical diagnosis:**
# - Missing cancer = no treatment → life-threatening
# - **High Recall is essential**
# 
# **False Positive (FP) is less critical:**
# - Unnecessary stress and tests, but not life-threatening
# 
# ---
# 
# ### Choosing the Right Metric
# 
# - **Accuracy:** When classes are balanced
# - **Precision:** When False Positives are costly
# - **Recall:** When False Negatives are costly (cancer detection!)
# - **F1-Score:** When you need balance
# 
# ---
# 
# ### Pros & Cons
# 
# **Advantages:**
# - Shows exactly where model fails
# - Multiple metrics for different needs
# - Easy to visualize with heatmap
# 
# **Disadvantages:**
# - Accuracy alone can be misleading
# - Requires understanding error trade-offs


# ============================
# Using same preprocessing from Decision Tree section above
# ============================

# Import data
tumors_data_dc = pd.read_csv("/kaggle/input/tumors-dataset/data.csv")
tumors_data_dc.drop(["Unnamed: 32","id"],axis = 1, inplace = True,errors='ignore')

# Classification
tumors_data_dc["diagnosis"] = [1 if each == "M" else 0 for each in tumors_data_dc.diagnosis]

# Split data
y_dc = tumors_data_dc.diagnosis.values
x_dc = tumors_data_dc.drop(["diagnosis"],axis = 1)

# Normalization
x_norm_dc = (x_dc - x_dc.min()) / (x_dc.max() - x_dc.min())  # Min-Max normalization

# Train, test, split - stratify maintains class balance
x_train_dc, x_test_dc, y_train_dc, y_test_dc = train_test_split(
    x_norm_dc, y_dc, test_size=0.15, random_state=42, stratify=y_dc
)

# Random Forest
rf = RandomForestClassifier(n_estimators = 100,random_state=42)
rf.fit(x_train_dc,y_train_dc)

# Confusion Matrix
y_pred = rf.predict(x_test_dc)
y_true = y_test_dc
cm = confusion_matrix(y_true,y_pred)

# Plot Heatmap
f , ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True , linewidths = 0.5,linecolor = "red", fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))

# 🎯 Model Comparison Summary

models_summary = {
    'Model': ['Logistic Regression', 'KNN', 'SVM', 'Naive Bayes', 'Decision Tree', 'Random Forest'],
    'Accuracy': [
        logistic_reg.score(x_test.T, y_test.T) * 100,
        test_scores[best_k-1] * 100,
        svm.score(x_test_svm, y_test_svm) * 100,
        nbc.score(x_test_nbc, y_test_nbc) * 100,
        dc.score(x_test_dc, y_test_dc) * 100,
        rf.score(x_test_dc, y_test_dc) * 100
    ]
}

df_summary = pd.DataFrame(models_summary)
df_summary = df_summary.sort_values('Accuracy', ascending=False).reset_index(drop=True)

# Plot comparison
fig = px.bar(df_summary, 
             x='Model', 
             y='Accuracy',
             title='Classification Models Accuracy Comparison',
             color='Accuracy',
             color_continuous_scale='Viridis')

fig.update_layout(xaxis_tickangle=-45)
fig.show()

print(df_summary)

# # 🎯 Key Takeaways
# 
# ## 📊 Quick Summary (TL;DR)
# **Best Model:** Random Forest (98.84% accuracy)  
# **Key Achievement:** Only 1 cancer missed out of 32, with 0 false alarms  
# **Top Features:** area_worst (15.8%), perimeter_worst (9.8%), concave points_worst (9.7%)  
# **Optimal KNN K:** 9 (97.66% accuracy)  
# **Critical Insight:** "Worst" measurements are stronger predictors than mean values
# 
# ---
# 
# ## 🔍 Detailed Analysis
# 
# ### Performance Metrics
# - **Random Forest Confusion Matrix:**
#   - ✅ True Negatives: 54 (all benign identified)
#   - ✅ False Positives: 0 (no false alarms)
#   - ⚠️ False Negatives: 1 (1 cancer missed)
#   - ✅ True Positives: 31 (97% Recall)
# 
# **Why This Matters:**
# - Recall for Malignant: 97% - Caught 31 of 32 cancers
# - Precision for Malignant: 100% - Every cancer prediction was correct
# - Ideal balance for medical diagnosis
# 
# ### Top Predictive Features
# | Rank | Feature | Importance |
# |------|---------|------------|
# | 1 | area_worst | 15.8% |
# | 2 | perimeter_worst | 9.8% |
# | 3 | concave points_worst | 9.7% |
# | 4 | concave points_mean | 9.6% |
# | 5 | perimeter_mean | 7.6% |
# 
# **Insight:** Extreme tumor characteristics (worst values) are better cancer indicators than averages.
# 
# ### Model Comparison
# | Model | Accuracy | Best For |
# |-------|----------|----------|
# | Random Forest | 98.84% | Maximum accuracy + interpretability |
# | SVM | 98.83% | High-dimensional medical data |
# | Logistic Regression | 98.25% | Fast, interpretable baseline |
# | KNN (K=9) | 97.66% | Quick deployment |
# | Decision Tree | 96.51% | Transparent rules |
# | Naive Bayes | 93.57% | Limited resources |
# 
# ### When to Use Which?
# - **Random Forest:** Production deployment (best overall)
# - **SVM:** Validation/backup system
# - **Logistic Regression:** Clinical explanations
# - **KNN:** Fast prototyping
# - **Decision Tree:** Educational purposes
# - **Naive Bayes:** Resource-constrained environments
# 
# ### Medical Insights
# 🎯 **For Cancer Detection:**
# - Prioritize Recall (catch all cancers) over Precision
# - 97% Recall = only 1 in 32 missed
# - 0 False Positives = no patient anxiety
# 
# ⚠️ **The One Miss:**
# - Highlights need for multiple diagnostic methods
# - Regular follow-ups essential
# - Lower thresholds for high-risk patients
# 
# ### Final Recommendation
# 1. Deploy Random Forest (98.84%, feature importance)
# 2. Use SVM as backup (98.83%)
# 3. Monitor False Negatives closely
# 4. Consider ensemble voting for reliability
# 5. Always combine with clinical judgment
# 
# **Bottom Line:** Random Forest wins for medical diagnosis - high accuracy, interpretability, and 97% Recall for cancer detection. 🏆


# # 🔗 References
# 
# ## 📚 My Machine Learning Series
# 
# This notebook is part of a comprehensive Machine Learning series:
# 
# | Notebook | Topics Covered |
# |----------|---------------|
# | 🎯 **Classification Models** | Logistic Regression, SVM, KNN, etc. *(Current)* |
# | 📈 **Regression Models** | [Link](https://www.kaggle.com/code/dandrandandran2093/machine-learning-regression-models) - Linear, Polynomial, Decision Tree, Random Forest |
# 
# ---
# 
# **Course:** Udemy - MACHINE LEARNING by DATAI TEAM
# 
# **Libraries:** NumPy, Pandas, Matplotlib, Plotly, Scikit-learn, Seaborn