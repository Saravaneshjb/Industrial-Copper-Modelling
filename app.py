import streamlit as st
from regression.reg_testing import RegTestingPipeline
from classification.class_testing import ClassificationTestingPipeline

# Streamlit App
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pages", ["Home", "Regression", "Classification"])

if page == "Home":
    st.title("Industrial Copper Modelling")
    st.write("""
        ### Description of the Problem Statement:
        The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.
        Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer. You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.
    """)

elif page == "Regression":
    st.title(" Industrial Copper Modelling - Regression")
    input_data = {
        'quantity tons': st.text_input("Quantity Tons (e.g., 54.1511)"),
        'customer': st.text_input("Customer (e.g., 30156308)"),
        'country': st.text_input("Country (e.g., 28)"),
        'application': st.text_input("Application (e.g., 10)"),
        'thickness': st.text_input("Thickness (e.g., 2)"),
        'width': st.text_input("Width (e.g., 1500)"),
        'product_ref': st.text_input("Product Ref (e.g., 1670798778)"),
        'item type': st.text_input("Item Type (e.g., W)"),
        'status': st.text_input("Status (e.g., Won)"),
        'item_date': st.text_input("Item Date (YYYYMMDD) (e.g., 20210401)"),
        'delivery_date': st.text_input("Delivery Date (YYYYMMDD) (e.g., 20210701)")
    }

    if st.button("Predict"):
        input_data['quantity tons'] = float(input_data['quantity tons'])
        input_data['customer'] = int(input_data['customer'])
        input_data['country'] = int(input_data['country'])
        input_data['application'] = int(input_data['application'])
        input_data['thickness'] = float(input_data['thickness'])
        input_data['width'] = float(input_data['width'])
        input_data['product_ref'] = int(input_data['product_ref'])
        # 'item type' and 'status' remain as strings
        input_data['item_date'] = input_data['item_date']
        input_data['delivery date'] = input_data['delivery date']

        pickle_folder_path = 'D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Industrial Copper Mining\\regression\\reg_pickle_files'
        test_pipeline = RegTestingPipeline(pickle_folder_path)
        preprocessed_df = test_pipeline.preprocess(input_data)
        predicted_selling_price = test_pipeline.predict(preprocessed_df)
        st.write(f"Predicted Selling Price: {predicted_selling_price}")

elif page == "Classification":
    st.title("Classification Model Input")
    input_data = {
        'quantity tons': st.text_input("Quantity Tons (e.g., 54.1511)"),
        'customer': st.text_input("Customer (e.g., 30156308)"),
        'country': st.text_input("Country (e.g., 28)"),
        'application': st.text_input("Application (e.g., 10)"),
        'thickness': st.text_input("Thickness (e.g., 2)"),
        'width': st.text_input("Width (e.g., 1500)"),
        'product_ref': st.text_input("Product Ref (e.g., 1670798778)"),
        'item type': st.text_input("Item Type (e.g., W)"),
        'item_date': st.text_input("Item Date (YYYYMMDD) (e.g., 20210401)"),
        'delivery date': st.text_input("Delivery Date (YYYYMMDD) (e.g., 20210701)"),
        'selling_price': st.text_input("Selling Price (e.g., 854)")
    }

    if st.button("Predict"):
        # Convert inputs to appropriate types
        input_data['quantity tons'] = float(input_data['quantity tons'])
        input_data['customer'] = int(input_data['customer'])
        input_data['country'] = int(input_data['country'])
        input_data['application'] = int(input_data['application'])
        input_data['thickness'] = float(input_data['thickness'])
        input_data['width'] = float(input_data['width'])
        input_data['product_ref'] = int(input_data['product_ref'])
        # 'item type' remains as string
        input_data['item_date'] = input_data['item_date']
        input_data['delivery date'] = input_data['delivery date']
        input_data['selling_price'] = float(input_data['selling_price'])

        pickle_folder_path = 'D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Industrial Copper Mining\\classification\\class_pickle_files'
        test_pipeline = ClassificationTestingPipeline(pickle_folder_path)
        preprocessed_df = test_pipeline.preprocess(input_data)
        prediction = test_pipeline.predict(preprocessed_df)
        st.write(f"Prediction: {prediction[0]}")