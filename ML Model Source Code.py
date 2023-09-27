# Importing libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, auc, f1_score
from sklearn.svm import SVC
from scipy.stats import skew
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file path to the Excel file
dataset_file_path = os.path.join(current_dir, "Data Source", "dataset information.xlsx")

# Loading dataset from Excel
dataset = pd.read_excel(dataset_file_path)

# Handling null values
print(dataset.info())

# Checking for NULLs in the data
print(dataset.isnull().sum())

# Encode the Class
encoding = {'High_risk': 1, 'Low_risk': 0}
dataset['Class'] = dataset['Class'].map(encoding)

# checking for skewness
skewness = skew(dataset)
print("Skewness:", skewness)

# clean up the column header
dataset.columns = dataset.columns.str.strip()
col_names = dataset.columns

# Checking for correlation
clean_table = dataset.drop(['participant number'], axis=1)
print(clean_table.corr()["Class"].sort_values(ascending=False))

# Output the descriptive summary
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
summary_stats = clean_table.describe()
print(summary_stats)

# preparing the features and target variables
X = dataset.drop(['Class'], axis=1)
X = X.drop(['participant number'], axis=1)
y = dataset['Class']

### NORMAL TEST TRAIN SPLIT
# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 0)

### K FOLD VALIDATION
# Define the number of folds
k = 10

# Initialize the KFold object
kf = KFold(n_splits=k, shuffle = True, random_state = 0)

# Create a class that will contain the performance score
class Result:
    def __init__(self, accuracy, auc, precision, recall, f1, err, fpr, tpr, roc_auc):
        self.accuracy = accuracy
        self.auc = auc
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.err = err
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        
# Define a function to calculate the mean score of each performance score        
def calculate_mean_attribute(result_list, attribute):
    attribute_sum = 0
    for result in result_list:
        attribute_sum += getattr(result, attribute)

    mean_attribute = attribute_sum / len(result_list)
    return mean_attribute

# Random Forest model
rf_model = RandomForestClassifier()
rf_scores = []

# SVM model
# default (C=1.0, kernel=rbf, gamma=auto)
svm_model = SVC() 
svm_scores = []

# KNN model
knn_model = KNeighborsClassifier()
knn_scores = []

# Iterate over the folds
for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Random Forest
    rf_model.fit(X_train, y_train)
    
    # SVM model
    svm_model.fit(X_train,y_train)
    
    # KNN model
    knn_model.fit(X_train, y_train)

    # Making predictions
    rf_predictions = rf_model.predict(X_test)
    knn_predictions = knn_model.predict(X_test)
    svm_predictions = svm_model.predict(X_test)

    # Evaluating model performance
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    rf_auc = roc_auc_score(y_test, rf_predictions)
    knn_auc = roc_auc_score(y_test, knn_predictions)
    svm_auc = roc_auc_score(y_test, svm_predictions)

    rf_precision = precision_score(y_test, rf_predictions)
    knn_precision = precision_score(y_test, knn_predictions)
    svm_precision = precision_score(y_test, svm_predictions)

    rf_recall = recall_score(y_test, rf_predictions)
    knn_recall = recall_score(y_test, knn_predictions)
    svm_recall = recall_score(y_test, svm_predictions)

    rf_f1 = f1_score(y_test, rf_predictions)
    knn_f1 = f1_score(y_test, knn_predictions)
    svm_f1 = f1_score(y_test, svm_predictions)

    rf_err = 1 - rf_accuracy
    knn_err = 1 - knn_accuracy
    svm_err = 1 - svm_accuracy

    # ROC Curve
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_predictions)
    knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_predictions)
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_predictions)

    rf_roc_auc = auc(rf_fpr, rf_tpr)
    knn_roc_auc = auc(knn_fpr, knn_tpr)
    svm_roc_auc = auc(svm_fpr, svm_tpr)
    
    rf_scores.append(Result(rf_accuracy, rf_auc, rf_precision, rf_recall, rf_f1, rf_err, rf_fpr, rf_tpr, rf_roc_auc))
    knn_scores.append(Result(knn_accuracy, knn_auc, knn_precision, knn_recall, knn_f1, knn_err, knn_fpr, knn_tpr, knn_roc_auc))
    svm_scores.append(Result(svm_accuracy, svm_auc, svm_precision, svm_recall, svm_f1, svm_err, svm_fpr, svm_tpr, svm_roc_auc))

# Printing the results
print("Random Forest Accuracy:", calculate_mean_attribute(rf_scores, "accuracy"))
print("SVM Accuracy:", calculate_mean_attribute(svm_scores, "accuracy"))
print("KNN Accuracy:", calculate_mean_attribute(knn_scores, "accuracy"))

print("Random Forest AUC:", calculate_mean_attribute(rf_scores, "auc"))
print("SVM AUC:", calculate_mean_attribute(svm_scores, "auc"))
print("KNN AUC:", calculate_mean_attribute(knn_scores, "auc"))

print("Random Forest Precision:", calculate_mean_attribute(rf_scores, "precision"))
print("SVM Precision:", calculate_mean_attribute(svm_scores, "precision"))
print("KNN Precision:", calculate_mean_attribute(knn_scores, "precision"))

print("Random Forest Recall:", calculate_mean_attribute(rf_scores, "recall"))
print("SVM Recall:", calculate_mean_attribute(svm_scores, "recall"))
print("KNN Recall:", calculate_mean_attribute(knn_scores, "recall"))

print("Random Forest F1 score:", calculate_mean_attribute(rf_scores, "f1"))
print("SVM F1 score:", calculate_mean_attribute(svm_scores, "f1"))
print("KNN F1 score:", calculate_mean_attribute(knn_scores, "f1"))

print("Random Forest ERR score:", calculate_mean_attribute(rf_scores, "err"))
print("SVM ERR score:", calculate_mean_attribute(svm_scores, "err"))
print("KNN ERR score:", calculate_mean_attribute(knn_scores, "err"))

# Plotting ROC Curve
import matplotlib.pyplot as plt

plt.figure()
plt.plot(
    calculate_mean_attribute(rf_scores, "fpr"), 
    calculate_mean_attribute(rf_scores, "tpr"), 
    label='Random Forest (AUC = %0.2f)' % 
    calculate_mean_attribute(rf_scores, "roc_auc"))

plt.plot(
    calculate_mean_attribute(svm_scores, "fpr"), 
    calculate_mean_attribute(svm_scores, "tpr"), 
    label='SVM (AUC = %0.2f)' % 
    calculate_mean_attribute(svm_scores, "roc_auc"))

plt.plot(
    calculate_mean_attribute(knn_scores, "fpr"), 
    calculate_mean_attribute(knn_scores, "tpr"), 
    label='KNN (AUC = %0.2f)' % 
    calculate_mean_attribute(knn_scores, "roc_auc"))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Random forest feature importance
feat_labels = X.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# Plot q227 v class
# distribution of scores for high and low risk
from matplotlib.colors import ListedColormap

q227 = sorted(dataset['Q227'].unique())
q227_highRisk = dataset[dataset['Class'] == 1]['Q227'].value_counts().sort_index()
q227_lowRisk = dataset[dataset['Class'] == 0]['Q227'].value_counts().sort_index()
df_counts = pd.DataFrame({'Q227': q227, 'High Risk': q227_highRisk, 'Low Risk': q227_lowRisk})
df_counts.fillna(0, inplace=True)
cMap = ListedColormap(['#0343df', '#e50000'])
ax = df_counts.plot.bar(x='Q227', colormap=cMap)
ax.set_xlabel(None)
ax.set_ylabel('Q227')
ax.set_title('Distribution of Scores of High and Low Risk Class for Q227')
plt.show()

# Plot q225 v class
# distribution of scores for high and low risk
from matplotlib.colors import ListedColormap

q225 = sorted(dataset['Q225'].unique())
q225_highRisk = dataset[dataset['Class'] == 1]['Q225'].value_counts().sort_index()
q225_lowRisk = dataset[dataset['Class'] == 0]['Q225'].value_counts().sort_index()
df_counts = pd.DataFrame({'Q225': q225, 'High Risk': q225_highRisk, 'Low Risk': q225_lowRisk})
df_counts.fillna(0, inplace=True)
cMap = ListedColormap(['#0343df', '#e50000'])
ax = df_counts.plot.bar(x='Q225', colormap=cMap)
ax.set_xlabel(None)
ax.set_ylabel('Q225')
ax.set_title('Distribution of Scores of High and Low Risk Class for Q225')
plt.show()
