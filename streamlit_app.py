import streamlit as st
import pandas as pd
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Memuat model dan scaler yang telah di-fit
model = load('best_trained_model.pkl')
standard_scaler = load('best_standard_scaler.pkl')

# Sidebar menu
st.sidebar.header('Menu')
menu = st.sidebar.selectbox('Pilih Menu', ['Prediction'])

if menu == 'Prediction':
    st.header('Prediksi Status Mahasiswa')
    
    def user_input_features():
        # Input data dari pengguna
        Application_order = st.slider('Application_order', 1, 10, 1)
        Previous_qualification_grade = st.slider('Previous_qualification_grade', 0.0, 190.0, 100.0)
        Admission_grade = st.slider('Admission_grade', 0.0, 200.0, 120.0)
        Curricular_units_1st_sem_grade = st.slider('Curricular_units_1st_sem_grade', 0.0, 20.0, 10.0)
        Curricular_units_2nd_sem_grade = st.slider('Curricular_units_2nd_sem_grade', 0.0, 20.0, 10.0)
        
        data = {
            'Application_order': Application_order,
            'Previous_qualification_grade': Previous_qualification_grade,
            'Admission_grade': Admission_grade,
            'Curricular_units_1st_sem_grade': Curricular_units_1st_sem_grade,
            'Curricular_units_2nd_sem_grade': Curricular_units_2nd_sem_grade,
        }
        features = pd.DataFrame(data, index=[0])
        return features
    
    df_input = user_input_features()
    
    # Menampilkan data input pengguna
    st.subheader('Input Data')
    st.write(df_input)
    
    # Daftar fitur numerik yang digunakan
    numerical_features = [
        'Application_order',
        'Previous_qualification_grade',
        'Admission_grade',
        'Curricular_units_1st_sem_grade',
        'Curricular_units_2nd_sem_grade'
    ]
    
    # ColumnTransformer untuk preprocessing
    preprocessor = ColumnTransformer(
        transformers=[('scaler', standard_scaler, numerical_features)],
        remainder='passthrough'
    )
    
    # Pipeline untuk preprocessing dan prediksi
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Fit pipeline menggunakan data pelatihan (hanya sekali selama pembuatan pipeline)
    # Pastikan model dan scaler sudah ter-fit pada data pelatihan sebelumnya

    try:
        # Prediksi menggunakan data input pengguna
        y_pred_test = pipeline.predict(df_input)
        
        # Menampilkan hasil prediksi
        st.subheader('Hasil Prediksi')
        status = 'Dropout' if y_pred_test[0] == 1 else 'Graduate'
        st.write(f"Status Mahasiswa: **{status}**")
    
    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")
