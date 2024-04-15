#Exploratory Data Analysis  

#Importing necessary packages for data exploration
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

#Function to read in dataset
def read_data(file_path):
    return pd.read_csv(file_path)

#Data exploration function
def explore_data(data):
    #Displaying basic information about the dataset:
    print('Dataset Columns & Variable Types: ', data.info)
    
    #Displaying first few rows of the dataset:
    print("\nFirst few rows of the Heart Disease Dataset:")
    print(data.head())
    
    #Using pandas .describe feature to provide descriptive statistics on dataset
    pd.set_option('display.max_columns', None)
    print("\nDescriptive Statistics: Heart Disease Dataset")
    print(data.describe())
    
    #Determining the data types within the dataset:
    print(data.dtypes)
    
    #Checking the dataset for any missing values:
    print("\nMissing Values:")
    print(data.isnull().sum())
    

#This function provides us with a range of visualised data exploration    
def visualise_exploration(data):    
    #Visulalising the distribution of missing values using missingno:
    plt.figure(figsize=(8, 4))
    msno.matrix(data)
    plt.title('Missing Value Distribution')
    plt.show()
    
    #Using matplotlib to explore and visualise the distribution of numerical features within the dataset
    num_features = data.select_dtypes(include=['float64', 'int64']).columns
    for feature in num_features:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()
        
    #Exploring the relationships between numerical features using pair plots
    # Split the numerical features into multiple subsets 
    num_feature_subsets = [num_features[i:i+3] for i in range(0, len(num_features), 3)]

    # Create pair plots for each subset
    for subset in num_feature_subsets:
        sns.pairplot(data[subset])
        plt.show()
    
    # Check the data type of 'target' variable
    target_dtype = data['target'].dtype
    print("Data type of 'target' variable:", target_dtype)
    
    # Convert 'target' variable to string or categorical if necessary
    if target_dtype != 'object':
        data['target'] = data['target'].astype(str)
    
    #Show the current data imbalance before any preprocessing steps are implemented:
    sns.countplot(x='target', data=data, hue='target', dodge=False)
    plt.title('Distribution of Target Variables within the Heart Dataset')
    plt.show()
    
    # Convert 'target' variable back to int64
    if data['target'].dtype != 'int64':
        data['target'] = data['target'].astype('int64')

    
