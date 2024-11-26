import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load model dan preprocessor
model = joblib.load('best_trained_model.pkl')
scaler = joblib.load('best_standard_scaler.pkl')

# Fungsi untuk memproses dan memprediksi data
def preprocess_and_predict(data):
    # Proses data sesuai dengan preprocessing yang sudah dilakukan sebelumnya
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction

# Aplikasi Streamlit
def main():
    st.title("Prediksi Dropout Mahasiswa")
    
    # Input dari pengguna (misalnya file CSV)
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Tampilkan data yang diupload
        st.write(df.head())
        
        # Lakukan prediksi
        predictions = preprocess_and_predict(df)
        
        # Tampilkan hasil prediksi
        st.write("Hasil Prediksi Status Dropout:")
        st.write(predictions)

        # Visualisasi tambahan jika diperlukan
        st.pyplot(plt)

if __name__ == '__main__':
    main()
