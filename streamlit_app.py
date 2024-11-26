import streamlit as st
import pandas as pd
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Memuat model dan scaler
model = load('best_trained_model.pkl')
standard_scaler = load('best_standard_scaler.pkl')

# Judul aplikasi
st.title("🚀 Prediksi Status Mahasiswa")

# Sidebar menu
st.sidebar.title("Navigasi")
menu = st.sidebar.radio('Pilih Menu:', ['Beranda', 'Prediksi'])

if menu == 'Beranda':
    st.header("Selamat Datang!")
    st.write("""
    Aplikasi ini dirancang untuk memprediksi status mahasiswa berdasarkan data akademik mereka. 
    Masukkan parameter mahasiswa di menu Prediksi untuk mengetahui hasilnya. 🎓
    """)

elif menu == 'Prediksi':
    st.header('🎯 Prediksi Status Mahasiswa')

    def input_form():
        with st.form(key='input_form'):
            st.subheader('Masukkan Parameter Mahasiswa')
            Application_order = st.slider('Urutan Pendaftaran', 1, 10, 1)
            Daytime_evening_attendance = st.selectbox('Kehadiran Siang/Malam', ['Siang', 'Malam'])
            Previous_qualification_grade = st.number_input('Nilai Kualifikasi Sebelumnya', 0.0, 200.0, 90.0, step=0.1)
            Admission_grade = st.number_input('Nilai Masuk', 0.0, 200.0, 90.0, step=0.1)
            Displaced = st.radio('Apakah Mahasiswa Tergusur?', ['Ya', 'Tidak'])
            Debtor = st.radio('Apakah Memiliki Hutang?', ['Ya', 'Tidak'])
            Tuition_fees_up_to_date = st.radio('Apakah Biaya Pendidikan Tepat Waktu?', ['Ya', 'Tidak'])
            Gender = st.radio('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
            Scholarship_holder = st.radio('Penerima Beasiswa?', ['Ya', 'Tidak'])
            Curricular_units_1st_sem_credited = st.slider('Mata Kuliah Semester 1 Diakui', 0, 30, 0)
            Curricular_units_1st_sem_enrolled = st.slider('Mata Kuliah Semester 1 Diikuti', 0, 30, 0)
            Curricular_units_1st_sem_approved = st.slider('Mata Kuliah Semester 1 Lulus', 0, 30, 0)
            Curricular_units_1st_sem_grade = st.slider('Nilai Semester 1', 0.0, 30.0, 0.0)
            Curricular_units_2nd_sem_credited = st.slider('Mata Kuliah Semester 2 Diakui', 0, 20, 0)
            Curricular_units_2nd_sem_enrolled = st.slider('Mata Kuliah Semester 2 Diikuti', 0, 20, 0)
            Curricular_units_2nd_sem_approved = st.slider('Mata Kuliah Semester 2 Lulus', 0, 20, 0)
            Curricular_units_2nd_sem_grade = st.slider('Nilai Semester 2', 0.0, 30.0, 0.0)
            submit = st.form_submit_button('Prediksi')
            
            if submit:
                return {
                    'Application_order': Application_order,
                    'Daytime_evening_attendance': 1 if Daytime_evening_attendance == 'Siang' else 0,
                    'Previous_qualification_grade': Previous_qualification_grade,
                    'Admission_grade': Admission_grade,
                    'Displaced': 1 if Displaced == 'Ya' else 0,
                    'Debtor': 1 if Debtor == 'Ya' else 0,
                    'Tuition_fees_up_to_date': 1 if Tuition_fees_up_to_date == 'Ya' else 0,
                    'Gender': 1 if Gender == 'Laki-laki' else 0,
                    'Scholarship_holder': 1 if Scholarship_holder == 'Ya' else 0,
                    'Curricular_units_1st_sem_credited': Curricular_units_1st_sem_credited,
                    'Curricular_units_1st_sem_enrolled': Curricular_units_1st_sem_enrolled,
                    'Curricular_units_1st_sem_approved': Curricular_units_1st_sem_approved,
                    'Curricular_units_1st_sem_grade': Curricular_units_1st_sem_grade,
                    'Curricular_units_2nd_sem_credited': Curricular_units_2nd_sem_credited,
                    'Curricular_units_2nd_sem_enrolled': Curricular_units_2nd_sem_enrolled,
                    'Curricular_units_2nd_sem_approved': Curricular_units_2nd_sem_approved,
                    'Curricular_units_2nd_sem_grade': Curricular_units_2nd_sem_grade,
                }
        return None

    input_data = input_form()

    if input_data:
        # Konversi data ke DataFrame
        df_input = pd.DataFrame(input_data, index=[0])

        # Menampilkan input pengguna
        st.subheader('📝 Data Masukan Mahasiswa')
        st.write(df_input)

        # Definisi fitur numerik
        numerical_features = [
            'Application_order', 'Previous_qualification_grade', 'Admission_grade',
            'Curricular_units_1st_sem_enrolled', 'Curricular_units_2nd_sem_enrolled',
            'Curricular_units_1st_sem_credited', 'Curricular_units_2nd_sem_credited',
            'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_approved',
            'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade'
        ]

        # Pipeline preprocessing dan prediksi
        preprocessor = ColumnTransformer(
            transformers=[('scaler', standard_scaler, numerical_features)],
            remainder='passthrough'
        )
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Prediksi
        y_pred_test = pipeline.predict(df_input)
        y_pred_proba = pipeline.predict_proba(df_input) if hasattr(model, 'predict_proba') else None

        # Menampilkan hasil prediksi
        st.subheader('🎓 Hasil Prediksi')
        status = 'Dropout' if y_pred_test[0] == 1 else 'Graduate'
        color = 'red' if status == 'Dropout' else 'green'

        if y_pred_proba is not None:
            proba = round(y_pred_proba[0][y_pred_test[0]], 3)
            st.markdown(f'<p style="color:{color}; font-size:24px;">{status} dengan probabilitas {proba}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color:{color}; font-size:24px;">{status} (Probabilitas tidak tersedia)</p>', unsafe_allow_html=True)
