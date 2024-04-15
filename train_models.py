#Model Training Module

#Within this module, all Machine Learning Models will be defined ready for use in the main code

#Importing necessary libraries & packages
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
import numpy as np

#KNN Creation & Training Function
def train_knn_best_k(x_train, y_train, max_k=30, cv=10):
    #Define a Dictionary for storage of K
    error_values = []
    #Start timer (measures training time)
    start_time = time.time()
    
    #Loop to find best K value
    for k in range(1, max_k + 1):
        model_knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model_knn, x_train, y_train, cv=cv)
        error_values.append(np.mean(scores))
        
    #Use best K to train model
    best_k = np.argmax(error_values) + 1
    model_knn = KNeighborsClassifier(n_neighbors=best_k)
    model_knn.fit(x_train, y_train)
    training_time = time.time() - start_time

    return model_knn, best_k, training_time

#LR Creation & Training Function
def train_logistic_regression_model(x_train, y_train):
    #Start timer (measures training time)
    start_time = time.time()
    # Create and train the logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    return lr_model, training_time

#FFN Creation & Training Function
def train_feedforward_nn_model(x_train, y_train, x_test, y_test, epochs=20, batch_size=32):
    #Start timer (measures training time)
    start_time = time.time()
    # Create the neural network model
    model_ffn = Sequential()
    model_ffn.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model_ffn.add(Dense(32, activation='relu'))
    model_ffn.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model_ffn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model_ffn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    
    #Calculate Training time
    training_time = time.time() - start_time
    
    return model_ffn, history, training_time

#GNB Creation & Training Function
def train_gaussian_nb_model(x_train, y_train):
    #Start timer
    start_time = time.time()
    
    # Create the Gaussian Naive Bayes model
    gnb_model = GaussianNB()

    # Train the model
    gnb_model.fit(x_train, y_train)
    
    #End timer
    training_time = time.time() - start_time

    return gnb_model, training_time


#Function to Create and Train RF Model
def train_rf_model(x_train, y_train):
    #Start timer
    start_time = time.time()
    
    #Create Random Forest model
    rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
    
    #Train RF model
    rf_model.fit(x_train, y_train)
    
    #End timer
    training_time = time.time() - start_time

    return rf_model, training_time

