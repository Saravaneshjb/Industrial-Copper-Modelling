import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import boxcox
import pickle

class TrainingPipeline:
    def __init__(self, filepath):
        self.filepath = filepath
        self.label_encoders = {}
        self.capping_bounds = {}
        self.boxcox_params = {}
        self.scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    
    def preprocess(self):
        # 1. Read the csv file into a dataframe
        copper_df = pd.read_excel(self.filepath)
        
        # 2. Drop the ID and market_reg features from Dataframe
        copper_df.drop(['id','material_ref'],axis=1,inplace=True)
        
        # 3. Create new feature 'Days_between_item_delivery'
        #Removing the '.0' from the dates
        copper_df['item_date']=copper_df['item_date'].apply(lambda x : str(x).split('.')[0])
        copper_df['delivery date']=copper_df['delivery date'].apply(lambda x : str(x).split('.')[0])

        # Deleting the two record which has incorrect item_date recored. 
        copper_df = copper_df[~copper_df['item_date'].isin(['19950000','20191919'])]

        # Deleting the one record which has incorrect delivery_date recored. 
        copper_df = copper_df[~copper_df['delivery date'].isin(['30310101','20212222'])]

        # converting the date to a particular format
        copper_df['item_date']=pd.to_datetime(copper_df['item_date'],format="%Y%m%d")
        copper_df['delivery date']=pd.to_datetime(copper_df['delivery date'],format="%Y%m%d")

        ## Calculating the no. of days between item_date and delivery date 
        copper_df['days_between_item_delivery']=abs((copper_df['item_date'] - copper_df['delivery date']).dt.days)
        

        ## 4. Dropping the item_date and delivery date features from teh dataframe 
        copper_df.drop(['item_date','delivery date'],axis=1,inplace=True)
        
        # 5. Nan value Imputation
        for features in copper_df.columns:
            if features in ['country','application','status','customer','days_between_item_delivery']:
                print(f"The Feature being processed is {features}")
                mode_value=copper_df[features].mode()[0]
                copper_df[features].fillna(mode_value, inplace=True)
            elif features in ['thickness','selling_price']:
                print(f"The Feature being processed is {features}")
                median_value=copper_df[features].median()
                copper_df[features].fillna(median_value, inplace=True)
            else:
                pass
        
        # 6. Convert Quantity tons to Numeric
        #Deleting the record which causes issue
        index_to_drop=copper_df[copper_df['quantity tons']=='e'].index
        copper_df.drop(index_to_drop, inplace=True)

        # Converting the feature to Numeric
        copper_df['quantity tons'] = pd.to_numeric(copper_df['quantity tons'], errors='coerce')
        copper_df.reset_index(drop=True, inplace=True)
        
        # 7. Feature Encoding
        for col in ['status', 'item type']:
            le = LabelEncoder()
            copper_df[col] = le.fit_transform(copper_df[col])
            self.label_encoders[col] = le
        
        # 8. Deleting records with negative values in selling price and 
        copper_df = copper_df[(copper_df['selling_price'] >= 0) & (copper_df['quantity tons'] >= 0)]
        
        # 9. Outlier Handling - Cap Method
        for col in ['thickness', 'width', 'quantity tons']:
            Q1 = copper_df[col].quantile(0.25)
            Q3 = copper_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            copper_df[col] = copper_df[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
            self.capping_bounds[col]=(lower_bound,upper_bound)
        
        # 10. Deleting Extreme values
        copper_df=copper_df[(copper_df['selling_price']>0) & (copper_df['selling_price']<=2000)]
        
        # 11. Apply Box Cox Transformation
        for col in ['thickness', 'width', 'quantity tons']:
            copper_df[col], param = boxcox(copper_df[col] + 1)  # Add 1 to avoid log(0)
            self.boxcox_params[col] = param
        
        # 12. Apply Standard Scaler
        copper_df[['thickness', 'width', 'quantity tons']] = self.scaler.fit_transform(copper_df[['thickness', 'width', 'quantity tons']])
        
        self.preprocessed_df = copper_df

        print("The shape of preprocessed dataframe :",copper_df.shape)
        print("The columns in preprocessed dataframe :",copper_df.columns)
        print(copper_df.head(1))
        
    def build_model(self):
        # 2. Split the dataframe into train and test
        X = self.preprocessed_df.drop('selling_price', axis=1)
        y = self.preprocessed_df['selling_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Build the Random Forest model
        self.rf_model.fit(X_train, y_train)
    
    def pickle_dump(self, obj, filename):
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
    
    def save_objects(self,folder_path):
        # Ensure the folder exists, if not, create it
        os.makedirs(folder_path, exist_ok=True)

        self.pickle_dump(self.capping_bounds, os.path.join(folder_path,'capping_bounds.pkl'))
        self.pickle_dump(self.label_encoders, os.path.join(folder_path,'label_encoders.pkl'))
        self.pickle_dump(self.scaler, os.path.join(folder_path,'scaler.pkl'))
        self.pickle_dump(self.boxcox_params, os.path.join(folder_path,'boxcox_params.pkl'))
        self.pickle_dump(self.rf_model, os.path.join(folder_path,'rf_model.pkl'))



if __name__ == "__main__":
    train_pipeline = TrainingPipeline('D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Industrial Copper Mining\\Copper_Set.xlsx')
    print("Starting the Preprocess workflow")
    train_pipeline.preprocess()
    print("Preprocess workflow - completed")
    print("Starting the model building workflow")
    train_pipeline.build_model()
    print("model building workflow-completed")

    # Specify the folder path where you want to save the pickle files
    pickle_path = 'D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Industrial Copper Mining\\regression\\reg_pickle_files'
    print("Starting Pickling workflow")
    train_pipeline.save_objects(pickle_path)
    print("Pickling workflow completed")