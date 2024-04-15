#Model testing & Evaluation Module

#Import necessary libraries & packages
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Function to test and evaluate each model
def evaluate_model(model, x_test, y_test, threshold=0.5):
    # Make predictions on the test set
    y_pred = model.predict(x_test)

    y_pred_binary = (y_pred > threshold).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred_binary)
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    classification_rep = classification_report(y_test, y_pred_binary)

    # Print or return any metrics you're interested in
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_mat)
    print("Classification Report:\n", classification_rep)

    return accuracy, confusion_mat, classification_rep
