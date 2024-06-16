# Industrial Copper Modelling Project - Regression & Classification Model

#### Problem Statement:
#### The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data. 
#### Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values."""

###### Link to the Dataset : https://docs.google.com/spreadsheets/d/18eR6DBe5TMWU9FnIewaGtsepDbV4BOyr/edit?gid=462557918#gid=462557918

### Download the dataset and store it in the project folder before proceeding

### Setting up the conda environment 
```conda create -p copperenv python==3.10```

### Activate the conda environment
```conda activate copperenv\```

### Install all the requirements 
```pip install -r requirements.txt```

### Regression Model & Classification Model 
#### Training - Regression 
#### Path : \Industrial Copper Mining\regression
```python reg_training.py```
#### Model Training would be completed and the following pickle files would be generated 
#### pickle file path : \Industrial Copper Mining\regression\reg_pickle_files
#### boxcox_params.pkl, capping_bounds.pkl, label_encoders.pkl, rf_model.pkl, scaler.pkl 

#### Training - Classification 
#### Path : \Industrial Copper Mining\classification
```python class_training.py```
#### Model Training would be completed and the following pickle files would be generated 
#### pickle file path : \Industrial Copper Mining\regression\class_pickle_files
#### boxcox_params.pkl, capping_bounds.pkl, label_encoders.pkl, rf_model.pkl, scaler.pkl


### Model Testing 
### Run the Streamlit app, pass the required inputs and click on Predict
### In order to test the Regression Model click on Regressoon
### In order to test the Classification Model click on Classification 
#### Path : \Industrial Opper Mining
```streamlit run app.py```

