#Data Preprocessing 

#Import necessary packages for data preprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#Data Cleaning
def clean_data(data):
    #Using nunique to check variable type and no of values stored in each:
    print(data.nunique())
    
    #Checking for character mistakes:
    #CA in the dataset is 0-3, however the use of nunique shows 0-4 & needs to be corrected:
    data['ca'].unique()
    print(data.ca.value_counts())
    #Finding rows that contain 4 and assigning a temp value
    data.loc[data['ca']==4, 'ca'] = np.NaN
    
    #Repeat process for thal inconsistency
    data['thal'].unique()
    print(data.thal.value_counts())
    data.loc[data['thal']==0, 'thal'] = np.NaN
    
    #Replace NaN with median: this method avoids data loss as we can fill mistakes rather than remove entire rows of data
    data = data.fillna(data.median())
    data.isnull().sum()
    
    #Check dataset for duplicate rows and remove
    data = data.drop_duplicates()
    
    return data

#This function checks the continous attributes for outliers and removes them
#This is an important step in preprocessing: outliers can skew the results when training models and lead to inconsistencies
def remove_outliers(data, attributes, drop=False):
    for attribute in attributes:
        attribute_data = data[attribute]
        Q1 = np.percentile(attribute_data, 25.) #Finding 25th percentile of the data for the given feature
        Q3 = np.percentile(attribute_data, 75.) #Finding 75th percentile of the data for the given feature
        IQR = Q3 - Q1                           #Calculating Interquartile range
        outlier_step = IQR * 1.5                #Defining Outlier step
        outliers = attribute_data[~((attribute_data >= Q1 - outlier_step) & (attribute_data <= Q3 + outlier_step))].index.tolist()
        
        if not drop:
            print(f"For the feature {attribute}, No of Outliers is {len(outliers)}")
        elif drop:
            data.drop(outliers, inplace=True, errors='ignore')
            print(f"Outliers from {attribute} feature removed")

    return data


#In this function, we will re-visualise the data imbalance after cleaning
def visualise_data_balance(data):
    
    # Convert 'target' variable to categorical if necessary
    if data['target'].dtype != 'category':
        data['target'] = data['target'].astype('category')
        
    sns.countplot(x='target', data=data, dodge=False)
    plt.title('Distribution of Target Variables within the Heart Dataset after Data Cleaning:')
    plt.show()
    
    # Convert 'target' variable back to int64
    if data['target'].dtype != 'int64':
        data['target'] = data['target'].astype('int64')
    
#Additional preprocessing steps to ensure that data is stored as correct type  
def preprocess_data(data):
    
    print(data.head())
    
    #separating dataset into categorical & numerical values
    continuous_attributes = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_attributes = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']


    #Changing necessary data types: categorical values should be stored as object
    data[categorical_attributes] = data[categorical_attributes].astype('object')
    

    return data
    



    


