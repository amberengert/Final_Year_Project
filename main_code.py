#Main Program: Comparative Analysis of Machine Learning Techniques for Heart Disease Prediction

#Pulling modules & functions created into main code:
from exploratory_analysis import read_data, explore_data, visualise_exploration
from data_preprocessing import clean_data, remove_outliers, visualise_data_balance, preprocess_data
from train_models import train_knn_best_k, train_logistic_regression_model, train_feedforward_nn_model, \
    train_gaussian_nb_model, train_rf_model
from test_evaluate_models import evaluate_model
from visualisation import visualise_correlation_matrix, plot_roc_curve, corr_matrix_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Importing and reading dataset from CSV file:
file_path = r'C:\Users\amber.engert\Downloads\University\Project\Data\heart.csv'
heart_data = read_data(file_path)

#DATA EXPLORATION - calling functions from exploratory_analysis.py
explore_data(heart_data)
visualise_exploration(heart_data)

#DATA CLEANING & PREPROCESSING
cleaned_data = clean_data(heart_data)
preprocessed_data = preprocess_data(cleaned_data)

#Outlier removal
continuous_attributes = ['age','trestbps','chol','thalach','oldpeak']
cleaned_data = remove_outliers(preprocessed_data, continuous_attributes)

#Visualising new data imbalace now data has been cleaned & preprocessed
visualise_data_balance(cleaned_data)

#VISUALIZATION
visualise_correlation_matrix(cleaned_data)

#Split the dataset into training and testing sets
features = cleaned_data.drop('target', axis=1)
target = cleaned_data['target']
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Implementing a standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Train KNN model
knn_model, training_time, best_k = train_knn_best_k(x_train, y_train)
knn_predictions = knn_model.predict(x_test)

#Train Logistic Regression model
lr_model, training_time = train_logistic_regression_model(x_train, y_train)
lr_predictions = lr_model.predict(x_test)

#Train Feedforward Neural Network model
ffn_model, training_time, ffn_history = train_feedforward_nn_model(x_train, y_train, x_test, y_test)
ffn_predictions = ffn_model.predict(x_test)

#Train Gaussian Naive Bayes model
gnb_model, training_time = train_gaussian_nb_model(x_train, y_train)
gnb_predictions = gnb_model.predict(x_test)

#Train Random Forest model
rf_model, training_time = train_rf_model(x_train, y_train)
rf_predictions = rf_model.predict(x_test)


#Evaluate the KNN model
start_time = time.time()
knn_accuracy, knn_confusion_mat, knn_classification_rep = evaluate_model(knn_model, x_test, y_test)
evaluation_time = time.time() - start_time  # Calculate evaluation time
print("KNN Model Evaluation:")
print("Accuracy:", knn_accuracy)
print("Confusion Matrix:\n", knn_confusion_mat)
print("Classification Report:\n", knn_classification_rep)
print("Training Time:", training_time)
print("Evaluation Time:", evaluation_time)
corr_matrix_eval(knn_model, x_test, y_test, 'KNN')

#Visualize ROC curve for KNN model
plot_roc_curve(knn_model, x_test, y_test, 'KNN')

#Evaluate the LR Model
start_time = time.time()
lr_accuracy, lr_confusion_mat, lr_classification_rep = evaluate_model(lr_model, x_test, y_test)
evaluation_time = time.time() - start_time  # Calculate evaluation time
print("LR Model Evaluation:")
print("Accuracy:", lr_accuracy)
print("Confusion Matrix:\n", lr_confusion_mat)
print("Classification Report:\n", lr_classification_rep)
print("Training Time:", training_time)
print("Evaluation Time:", evaluation_time)
corr_matrix_eval(lr_model, x_test, y_test, 'LR')

#Visualize ROC curve for LR model
plot_roc_curve(lr_model, x_test, y_test, 'LR')

#Evaluate the FFN Model
start_time = time.time()
ffn_accuracy, ffn_confusion_mat, ffn_classification_rep = evaluate_model(ffn_model, x_test, y_test)
evaluation_time = time.time() - start_time  # Calculate evaluation time
print("FFN Model Evaluation:")
print("Accuracy:", ffn_accuracy)
print("Confusion Matrix:\n", ffn_confusion_mat)
print("Classification Report:\n", ffn_classification_rep)
print("Training Time:", training_time)
print("Evaluation Time:", evaluation_time)
corr_matrix_eval(ffn_model, x_test, y_test, 'FFN')

#Visualize ROC curve for FFN model
plot_roc_curve(ffn_model, x_test, y_test, 'FFN')

#Evaluate the GNB Model
start_time = time.time()
gnb_accuracy, gnb_confusion_mat, gnb_classification_rep = evaluate_model(gnb_model, x_test, y_test)
evaluation_time = time.time() - start_time  # Calculate evaluation time
print("GNB Model Evaluation:")
print("Accuracy:", gnb_accuracy)
print("Confusion Matrix:\n", gnb_confusion_mat)
print("Classification Report:\n", gnb_classification_rep)
print("Training Time:", training_time)
print("Evaluation Time:", evaluation_time)
corr_matrix_eval(gnb_model, x_test, y_test, 'GNB')

#Visualize ROC curve for GNB model
plot_roc_curve(gnb_model, x_test, y_test, 'GNB')

#Evaluate the RF Model
start_time = time.time()
rf_accuracy, rf_confusion_mat, rf_classification_rep = evaluate_model(rf_model, x_test, y_test)
evaluation_time = time.time() - start_time  # Calculate evaluation time
print("RF Model Evaluation:")
print("Accuracy:", rf_accuracy)
print("Confusion Matrix:\n", rf_confusion_mat)
print("Classification Report:\n", rf_classification_rep)
print("Training Time:", training_time)
print("Evaluation Time:", evaluation_time)
corr_matrix_eval(rf_model, x_test, y_test, 'GNB')

#Visualize ROC curve for RF model
plot_roc_curve(rf_model, x_test, y_test, 'RF')

#Store results in a dictionary
model_results = {
    "KNN": knn_accuracy,
    "LR": lr_accuracy,
    "FFN": ffn_accuracy,
    "GNB": gnb_accuracy,
    "RF": rf_accuracy
}


#Find the model with the best accuracy
best_model = max(model_results, key=model_results.get)
best_accuracy = model_results[best_model]
print(f"The Model with the highest accuracy is {best_model} with a score of {best_accuracy:.2%}")

#Find the model with the worst accuracy
worst_model = min(model_results, key=model_results.get)
worst_accuracy = model_results[worst_model]
print(f"The Model with the lowest accuracy is {worst_model} with a score of {worst_accuracy:.2%}")