import streamlit as st
import pandas as pd
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Memuat model dan scaler
model = load('best_trained_model.pkl')
standard_scaler = load('best_standard_scaler.pkl')

# Judul aplikasi dengan ikon
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🎓 Prediksi Status Mahasiswa</h1>", unsafe_allow_html=True)

# Sidebar untuk menu dengan warna menarik
st.sidebar.markdown(
    "<h2 style='color: #117A65;'>📋 Menu</h2>", unsafe_allow_html=True
)
menu = st.sidebar.radio('Pilih Menu:', ['Prediksi Mahasiswa'])

if menu == 'Prediksi Mahasiswa':

    def user_input_features():
        col1, col2 = st.columns(2)

        with col1:
            Application_order = st.slider('Aplication Order', 1, 10, 1)
            Previous_qualification_grade = st.slider(
                'Nilai Kualifikasi Sebelumnya', 0.0, 190.0, 90.0)
            Admission_grade = st.slider(
                'Nilai Penerimaan', 0.0, 190.0, 90.0)

        with col2:
            Daytime_evening_attendance = st.selectbox(
                'Kehadiran (Siang/Malam)', ('Daytime', 'Evening'))
            Tuition_fees_up_to_date = st.selectbox(
                'Pembayaran Up-to-date', ('Yes', 'No'))
            Gender = st.selectbox('Jenis Kelamin', ('Male', 'Female'))
            Scholarship_holder = st.selectbox(
                'Penerima Beasiswa', ('Yes', 'No'))
            Displaced = st.selectbox('Mahasiswa Pindahan', ('Yes', 'No'))
            Debtor = st.selectbox('Status Debitur', ('Yes', 'No'))
            
        Curricular_units_1st_sem_enrolled = st.slider(
            'Mata Kuliah 1st Sem Diambil', 0, 30, 0)
        Curricular_units_1st_sem_approved = st.slider(
            'Mata Kuliah 1st Sem Disetujui', 0, 30, 0)
        Curricular_units_1st_sem_credited = st.slider(
                'SKS 1st Sem Diakui', 0, 30, 0)
        Curricular_units_1st_sem_grade = st.slider(
            'Nilai 1st Sem', 0.0, 30.0, 0.0)
        Curricular_units_2nd_sem_enrolled = st.slider(
            'Mata Kuliah 2nd Sem Diambil', 0, 20, 0)
        Curricular_units_2nd_sem_approved = st.slider(
            'Mata Kuliah 2nd Sem Disetujui', 0, 20, 0)
        Curricular_units_2nd_sem_credited = st.slider(
                'SKS 2nd Sem Diakui', 0, 20, 0)
        Curricular_units_2nd_sem_grade = st.slider(
            'Nilai 2nd Sem', 0.0, 30.0, 0.0)

        data = {
            'Application_order': Application_order,
            'Daytime_evening_attendance': 1 if Daytime_evening_attendance == 'Daytime' else 0,
            'Previous_qualification_grade': Previous_qualification_grade,
            'Admission_grade': Admission_grade,
            'Displaced': 1 if Displaced == 'Yes' else 0,
            'Debtor': 1 if Debtor == 'Yes' else 0,
            'Tuition_fees_up_to_date': 1 if Tuition_fees_up_to_date == 'Yes' else 0,
            'Gender': 1 if Gender == 'Male' else 0,
            'Scholarship_holder': 1 if Scholarship_holder == 'Yes' else 0,
            'Curricular_units_1st_sem_credited': Curricular_units_1st_sem_credited,
            'Curricular_units_1st_sem_enrolled': Curricular_units_1st_sem_enrolled,
            'Curricular_units_1st_sem_approved': Curricular_units_1st_sem_approved,
            'Curricular_units_1st_sem_grade': Curricular_units_1st_sem_grade,
            'Curricular_units_2nd_sem_credited': Curricular_units_2nd_sem_credited,
            'Curricular_units_2nd_sem_enrolled': Curricular_units_2nd_sem_enrolled,
            'Curricular_units_2nd_sem_approved': Curricular_units_2nd_sem_approved,
            'Curricular_units_2nd_sem_grade': Curricular_units_2nd_sem_grade,
        }
        return pd.DataFrame(data, index=[0])

    df_input = user_input_features()

    st.markdown("<h3 style='color: #1F618D;'>📊 Data Input Mahasiswa</h3>", unsafe_allow_html=True)
    st.write(df_input)

    # Define categorical features
    numerical_features = ['Application_order', 'Previous_qualification_grade', 'Admission_grade',
                          'Curricular_units_1st_sem_enrolled', 'Curricular_units_2nd_sem_enrolled',
                          'Curricular_units_1st_sem_credited', 'Curricular_units_2nd_sem_credited',
                          'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_approved',
                          'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade']

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', standard_scaler, numerical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.named_steps['preprocessor'].fit(df_input)
    y_pred_test = pipeline.predict(df_input)

    st.markdown("<h3 style='color: #1F618D;'>🔍 Hasil Prediksi</h3>", unsafe_allow_html=True)
    status = 'Dropout' if y_pred_test[0] == 1 else 'Graduate'
    color = 'red' if status == 'Dropout' else 'green'

    if hasattr(model, 'predict_proba'):
        y_pred_proba = pipeline.predict_proba(df_input)
        proba = round(y_pred_proba[0][y_pred_test[0]], 3)
        st.markdown(f"<h4 style='color:{color};'>📝 {status} dengan probabilitas {proba}</h4>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h4 style='color:{color};'>📝 {status} (Probabilitas tidak tersedia)</h4>", unsafe_allow_html=True)
