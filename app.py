import streamlit as st
import pandas as pd
import joblib

# Load your saved machine learning model
@st.cache_resource
def load_model():
    return joblib.load('car_model.joblib')

model = load_model()

# Title of your web app
st.title("🚗 Car Price Prediction App")
st.write("Enter your car's details below to predict its market price!")

# Input fields for numerical features
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2015, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=40000, step=500)
mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=18.5, step=0.1)
engine = st.number_input("Engine Capacity (CC)", min_value=500, max_value=8000, value=1248, step=10)
power = st.number_input("Power (bhp)", min_value=20.0, max_value=600.0, value=88.0, step=1.0)
seats = st.number_input("Seats", min_value=2, max_value=10, value=5, step=1)

# Dropdowns for categorical features
fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
location = st.selectbox("Location", ["Mumbai", "Pune", "Chennai", "Coimbatore", "Hyderabad", "Kochi", "Kolkata", "Delhi", "Jaipur", "Bangalore", "Ahmedabad"])

# Brand selection
brands_list = [
    "Maruti", "Hyundai", "Honda", "Toyota", "Mahindra", "Ford", "Volkswagen", "Chevrolet", 
    "Audi", "BMW", "Nissan", "Skoda", "Renault", "Fiat", "Tata", "Mercedes-Benz", "Mitsubishi", 
    "Land", "Jaguar", "Mini", "Volvo", "Porsche", "Jeep", "ISUZU", "Smart", "Bentley", "Lamborghini", "Force"
]
selected_brand = st.selectbox("Car Brand", brands_list)

# Prediction button
if st.button("Predict Price", type="primary"):
    try:
        # Dynamically fetch the exact feature names your trained model expects
        model_features = model.feature_names_in_
        
        # Create a dataframe filled with zeros matching the model's training columns
        input_data = pd.DataFrame(0, index=[0], columns=model_features)
        
        # Assign numerical values safely using exact column names from your notebook
        if 'Year' in input_data.columns: input_data['Year'] = year
        if 'Kilometers_Driven' in input_data.columns: input_data['Kilometers_Driven'] = km_driven
        if 'Mileage' in input_data.columns: input_data['Mileage'] = mileage
        if 'Engine' in input_data.columns: input_data['Engine'] = engine
        if 'Power' in input_data.columns: input_data['Power'] = power
        if 'Seats' in input_data.columns: input_data['Seats'] = seats
        
        # Assign one-hot encoded categorical variables
        loc_col = f"Location_{location}"
        if loc_col in input_data.columns: input_data[loc_col] = 1
        
        fuel_col = f"Fuel_Type_{fuel_type}"
        if fuel_col in input_data.columns: input_data[fuel_col] = 1
        
        if transmission == "Manual" and "Transmission_Manual" in input_data.columns:
            input_data["Transmission_Manual"] = 1
            
        owner_col = f"Owner_Type_{owner_type}"
        if owner_col in input_data.columns: input_data[owner_col] = 1
        
        brand_col = f"Brand_{selected_brand}"
        if brand_col in input_data.columns: input_data[brand_col] = 1
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Display the result
        st.success(f"Estimated Car Price: **₹ {prediction[0]:,.2f} Lakhs**")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")