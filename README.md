# iris-KNN-classifier

# Iris Species Classification using K-Nearest Neighbors (KNN)

This project demonstrates how to perform exploratory data analysis (EDA) and build a K-Nearest Neighbors (KNN) model to classify Iris flower species based on their sepal and petal measurements. The dataset used is `Iris-Copy1.csv`, which contains 150 entries with four features and a target variable (`Species`).

---

## **Table of Contents**
1. [Prerequisites](#prerequisites)
2. [Getting Started](#getting-started)
3. [Running the Code](#running-the-code)
4. [Code Explanation](#code-explanation)
5. [Results](#results)
6. [License](#license)

---

## **Prerequisites**
Before running the code, ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  ```bash
  pip install numpy pandas seaborn matplotlib scikit-learn
  ```
- Jupyter Notebook (optional, for running `.ipynb` files).

---

## **Getting Started**
1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/iris-KNN-classifier.git
   cd iris-KNN-classifier
   ```

2. **Download the Dataset**  
   Ensure the dataset `Iris-Copy1.csv` is in the same directory as the script or notebook.

---

## **Running the Code**
1. **Using Jupyter Notebook**  
   - Open the `.ipynb` file in Jupyter Notebook.
   - Run each cell sequentially to execute the code.

2. **Using Python Script**  
   - Save the code in a `.py` file (e.g., `iris_classification.py`).
   - Run the script using:
     ```bash
     python iris_classification.py
     ```

---

## **Code Explanation**
### **1. Import Libraries**
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```
- Libraries used for data manipulation, visualization, and modeling.

### **2. Load and Explore Data**
```python
data = pd.read_csv('Iris-Copy1.csv')
data.head()
data.shape
data.describe()
data.info()
data.isnull().sum()
```
- Load the dataset and explore its structure, summary statistics, and missing values.

### **3. Data Visualization**
```python
sns.heatmap(data.corr(), annot=True, cmap='Dark2')
sns.pairplot(data, hue='Species')
sns.boxplot(data['SepalLengthCm'])
sns.boxplot(data['SepalWidthCm'])
```
- Visualize correlations, pair plots, and box plots to understand the data distribution.

### **4. Prepare Data for Modeling**
```python
x = data.drop(columns='Species')
y = data['Species']
```
- Separate the features (`x`) and target variable (`y`).

### **5. Train-Test Split**
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
- Split the data into training and testing sets.

### **6. Build and Train Model**
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
```
- Train a KNN model on the training data.

### **7. Evaluate Model**
```python
y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
```
- Evaluate the model using accuracy, confusion matrix, and classification report.

---

## **Results**
- **Accuracy**: The model achieved an accuracy of **96.67%** on the test set.
- **Confusion Matrix**:
  ```
  [[11,  0,  0],
   [ 0, 12,  1],
   [ 0,  0,  6]]
  ```
- **Classification Report**:
  ```
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       1.00      0.92      0.96        13
 Iris-virginica       0.86      1.00      0.92         6

       accuracy                           0.97        30
      macro avg       0.95      0.97      0.96        30
   weighted avg       0.97      0.97      0.97        30
  ```

---

## **License**
This project is open-source and available under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as needed.

---

## **Support**
If you encounter any issues or have questions, feel free to open an issue in this repository or contact me at [your-email@example.com](mailto:your-email@example.com).

---

Enjoy exploring the Iris species classification model! 🌸
