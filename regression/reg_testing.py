import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.special import boxcox1p
import pickle



class RegTestingPipeline:
    def __init__(self, pickle_folder_path):
        self.pickle_folder_path=pickle_folder_path
        self.feature_order = ['quantity tons', 'customer', 'country', 'status', 'item type',
       'application', 'thickness', 'width', 'product_ref',
       'days_between_item_delivery']
        self.capping_bounds = self.pickle_load(os.path.join(self.pickle_folder_path, 'capping_bounds.pkl'))
        self.label_encoders = self.pickle_load(os.path.join(self.pickle_folder_path, 'label_encoders.pkl'))
        self.scaler = self.pickle_load(os.path.join(self.pickle_folder_path, 'scaler.pkl'))
        self.boxcox_params = self.pickle_load(os.path.join(self.pickle_folder_path, 'boxcox_params.pkl'))
        self.rf_model = self.pickle_load(os.path.join(self.pickle_folder_path, 'rf_model.pkl'))
    
    def preprocess(self, input_data):
        df = pd.DataFrame([input_data])
        
        # 3. Convert item_date and delivery date into "Days_between_item_delivery"
        # converting the date to a particular format
        df['item_date']=pd.to_datetime(df['item_date'],format="%Y%m%d")
        df['delivery date']=pd.to_datetime(df['delivery date'],format="%Y%m%d")
        df['days_between_item_delivery'] = (pd.to_datetime(df['delivery date']) - pd.to_datetime(df['item_date'])).dt.days
        #Dropping the item_date and delivery date features from teh dataframe 
        df.drop(['item_date','delivery date'],axis=1,inplace=True)
        
        # 4. Label Encoding
        for col in ['status', 'item type']:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col])
        
        # 5. Outlier Capping
        for col in ['thickness', 'width', 'quantity tons']:
            lower_bound, upper_bound = self.capping_bounds[col]
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        # 6. Apply Box Cox Transformation
        for col in ['thickness', 'width', 'quantity tons']:
            param = self.boxcox_params[col]
            df[col] = boxcox1p(df[col], param)
        
        # 7. Apply Standard Scaler
        df[['thickness', 'width', 'quantity tons']] = self.scaler.transform(df[['thickness', 'width', 'quantity tons']])
        
        #Ensure the features are in the same order as training
        df=df[self.feature_order]
        # print(df)
        
        return df
    
    def predict(self, preprocessed_df):
        return self.rf_model.predict(preprocessed_df)
    
    def pickle_load(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


# if __name__ == "__main__":
#     # Usage
#     input_data = {
#         'quantity tons': 54.1511,
#         'customer': 30156308,
#         'country': 28,
#         'application': 10,
#         'thickness': 2,
#         'width': 1500,
#         'product_ref': 1670798778,
#         'item type': 'W',
#         'status': 'Won',
#         'item_date': '20210401',
#         'delivery date': '20210701'
#     }


#     pickle_folder_path = 'D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Industrial Copper Mining\\regression\\reg_pickle_files'
#     test_pipeline = RegTestingPipeline(pickle_folder_path)
#     print("Initiated Preprocess ")
#     preprocessed_df = test_pipeline.preprocess(input_data)
#     print("Preprocessing Completed")
#     print("Initiated Prediction workflow")
#     predicted_selling_price = test_pipeline.predict(preprocessed_df)
#     print("Prediction workflow Completed")
#     print(f"Predicted Selling Price: {predicted_selling_price}")
