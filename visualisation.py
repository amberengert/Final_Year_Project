#Module for Visualisation of relevant graphs

#Import necessary libraries & packages 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score

#Function to visualise correlation matrices
def visualise_correlation_matrix(data):
    A = data.drop('target', axis=1)  
    b = data['target']  

    #Concatenating the features and target variable to create a new DataFrame for the correlation matrix
    data_combined = pd.concat([A, b], axis=1)

    #Calculate the correlation matrix
    correlation_matrix = data_combined.corr()

    #Display the correlation of each feature with the target variable, and heatmap for visulatisation
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='PiYG', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap - Heart Disease')
    plt.show()


#Function to plot ROC curves and Calculate AUC Scores
def plot_roc_curve(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

#Function to create confusion matrices for each model
def corr_matrix_eval(model, x_test, y_test, model_name):

    y_pred = model.predict(x_test)
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Reshape y_pred_binary to make it 1-dimensional
    y_pred_binary = y_pred_binary.flatten()


    #Create a DataFrame with true labels and predicted labels
    correlation_data = pd.DataFrame({'True_Labels': y_test, 'Predicted_Labels': y_pred_binary})

    # Create a confusion matrix using crosstab
    confusion_matrix = pd.crosstab(correlation_data['True_Labels'], correlation_data['Predicted_Labels'], rownames=['True'], colnames=['Predicted'])

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

