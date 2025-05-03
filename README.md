## Student Dropout Prediction at Jaya Jaya Institute Using Random Forest

### **Business Understanding**
Jaya Jaya Institut adalah sebuah institusi pendidikan terkemuka yang telah beroperasi sejak tahun 2000 dan memiliki reputasi yang sangat baik dalam mencetak lulusan berkualitas tinggi. Meskipun memiliki prestasi yang gemilang, institusi ini kini menghadapi tantangan serius dengan tingginya angka dropout siswa, yang dapat merusak reputasi dan efisiensi operasionalnya. Tingkat dropout yang mencapai 32.12% menjadi masalah utama yang perlu segera diatasi.

Dengan lebih memahami faktor-faktor yang mempengaruhi dropout siswa, Jaya Jaya Institut berencana untuk menggunakan data analitik untuk mengidentifikasi siswa yang berpotensi dropout lebih awal. Hal ini memungkinkan pihak institusi untuk memberikan bimbingan, dukungan, dan intervensi yang diperlukan guna meningkatkan retensi siswa. Dengan upaya ini, diharapkan angka dropout dapat dikurangi, yang pada gilirannya akan membantu meningkatkan kelulusan siswa, mempertahankan reputasi institusi, serta meningkatkan kepuasan dan keberhasilan akademik siswa.

### **Permasalahan Bisnis**
1. **Tingkat Dropout yang Tinggi:** Saat ini, sekitar 32.12% siswa mengalami *dropout*, yang menghambat pertumbuhan dan keberlanjutan perusahaan.
2. **Rendahnya Tingkat Keberhasilan Ujian:** Dengan tingkat keberhasilan ujian yang hanya 55.87%, banyak siswa yang tidak lulus ujian atau tidak dapat menyelesaikan program pendidikan.
3. **Faktor Demografis yang Mempengaruhi Dropout:** Terdapat ketidakseimbangan antara jumlah siswa laki-laki dan perempuan yang menyebabkan ketidakstabilan dalam tingkat *dropout*.
4. **Tantangan Beasiswa:** Siswa yang tidak menerima beasiswa menunjukkan kecenderungan *dropout* yang lebih tinggi dibandingkan dengan mereka yang menerima beasiswa.

### **Cakupan Proyek**
Proyek ini bertujuan untuk:
1. Menganalisis data demografis siswa untuk mengidentifikasi faktor-faktor yang berkontribusi terhadap *dropout*.
2. Mengembangkan model prediksi menggunakan algoritma Machine Learning untuk memprediksi kemungkinan seorang siswa akan *dropout*.
3. Membuat dashboard interaktif yang dapat digunakan oleh manajemen untuk melacak KPI terkait siswa, seperti tingkat *dropout*, keberhasilan ujian, dan distribusi siswa berdasarkan beasiswa.
4. Memberikan rekomendasi berbasis data untuk meningkatkan retensi siswa dan meningkatkan tingkat kelulusan.

### **Persiapan**
**Sumber data:**
Data yang digunakan dalam proyek ini berasal dari database [dicoding](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md) yang mencakup informasi terkait status siswa, distribusi program studi, usia, jenis kelamin, beasiswa, nilai ujian, dan data demografis lainnya.

**Setup Environment:**
1. **Pastikan Python Terinstal:** Gunakan Python 3.x.
2. **Install Dependensi:**
   Instal dependensi yang diperlukan menggunakan:
   ```
   pip install -r requirements.txt
   ```
3. **Jalankan Jupyter Notebook:** Untuk melakukan eksplorasi data dan membangun model.
   ```
   jupyter notebook
   ```

### **Business Dashboard**
Dashboard bisnis yang telah dibuat di **Looker Studio** menyajikan visualisasi data yang mencakup:
- **Jumlah Total Siswa:** 4,424
- **Tingkat Dropout:** 32.12%
- **Tingkat Keberhasilan Ujian:** 55.87%
- **Distribusi Gender dan Program Studi:** Menampilkan proporsi siswa berdasarkan gender dan program studi.
- **Distribusi Dropout Berdasarkan Gender dan Usia:** Untuk memahami hubungan antara usia dan gender terhadap tingkat *dropout*.
- **Rata-rata Nilai Penerimaan:** Distribusi rata-rata nilai penerimaan siswa yang terdaftar, lulus, dan *dropout*.
- **Siswa Beasiswa vs Non-beasiswa:** Menampilkan perbandingan siswa yang menerima beasiswa dengan yang tidak.
- **Status Akademik vs Dropout:** Membantu memantau perkembangan dan tingkat keberhasilan akademik.

**Link untuk mengakses dashboard:**  
[Klik di sini untuk mengakses dashboard di Looker Studio](https://lookerstudio.google.com/reporting/0df09394-6cad-4720-aca2-e193843e3b34)

### **Menjalankan Sistem Machine Learning**
Berikut adalah langkah-langkah yang lebih terperinci untuk menjalankan sistem prediksi Machine Learning menggunakan algoritma Random Forest:

1. **Buka terminal atau command prompt**.
2. Ketik perintah `jupyter notebook` dan tekan **Enter**. Ini akan membuka antarmuka Jupyter Notebook di browser web Anda.
3. Pilih file notebook yang ingin Anda gunakan untuk melanjutkan langkah-langkah berikut.

#### **Langkah 2: Lakukan Analisis Data Eksploratif (EDA)**
1. **Baca dataset** yang Anda miliki ke dalam notebook, misalnya:
   ```python
   import pandas as pd
   data = pd.read_csv('data_siswa.csv')
   ```
2. **Eksplorasi data** untuk memahami struktur dan pola penting, seperti:
   - Menampilkan beberapa baris pertama dataset: `data.head()`
   - Menyusun statistik deskriptif: `data.describe()`
   - Memeriksa nilai yang hilang: `data.isnull().sum()`
3. **Identifikasi atribut** yang dapat mempengaruhi kemungkinan dropout, seperti data demografis dan akademik (misalnya, usia, nilai rata-rata, jenis kelamin, status beasiswa, dll.).

#### **Langkah 3: Skalasi Atribut dengan StandardScaler**
1. **Identifikasi atribut numerik** yang memiliki rentang nilai yang berbeda dan memerlukan skala.
2. Gunakan `StandardScaler` dari `scikit-learn` untuk menormalkan data numerik:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   data[['age', 'avg_grade']] = scaler.fit_transform(data[['age', 'avg_grade']])
   ```

#### **Langkah 4: Buat Model dengan Random Forest**
1. **Pisahkan data menjadi fitur dan target**. Misalnya, target adalah status siswa (graduate, dropout, enrolled):
   ```python
   X = data.drop('status', axis=1)  # Fitur
   y = data['status']  # Target
   ```
2. **Bagi data menjadi set pelatihan dan pengujian**:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   ```
3. **Buat model Random Forest** menggunakan `RandomForestClassifier` dari `scikit-learn`:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   ```
4. **Evaluasi model** menggunakan data pengujian:
   ```python
   from sklearn.metrics import accuracy_score, f1_score
   y_pred = model.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   f1 = f1_score(y_test, y_pred, average='weighted')
   print(f'Accuracy: {accuracy:.2f}')
   print(f'F1 Score: {f1:.2f}')
   ```

5. **Cek fitur penting** untuk memahami faktor-faktor yang paling berpengaruh pada prediksi dropout:
   ```python
   feature_importance = model.feature_importances_
   print(feature_importance)
   ```

6. **Simpan model dan encoder** untuk digunakan nantinya:
   ```python
   import joblib
   joblib.dump(model, 'random_forest_model.pkl')
   joblib.dump(scaler, 'scaler.pkl')
   ```

#### **Langkah 5: Buat Streamlit GUI untuk Klasifikasi Siswa**
1. **Buat file Python baru** untuk aplikasi Streamlit:
   ```python
   import streamlit as st
   import pandas as pd
   import joblib
   from sklearn.preprocessing import StandardScaler

   # Muat model dan encoder
   model = joblib.load('random_forest_model.pkl')
   scaler = joblib.load('scaler.pkl')

   # Antarmuka pengguna
   st.title("Prediksi Dropout Siswa")
   age = st.number_input("Masukkan Usia Siswa:")
   avg_grade = st.number_input("Masukkan Nilai Rata-Rata Siswa:")
   
   # Proses input dan prediksi
   input_data = scaler.transform([[age, avg_grade]])
   prediction = model.predict(input_data)
   
   if prediction == 0:
       st.write("Prediksi: Lulus")
   elif prediction == 1:
       st.write("Prediksi: Dropout")
   else:
       st.write("Prediksi: Terdaftar")
   ```

#### **Langkah 6: Jalankan Streamlit**
1. **Jalankan perintah Streamlit** di terminal atau command prompt:
   ```bash
   streamlit run app.py
   ```

#### **Deploy Aplikasi Streamlit**
Aplikasi Streamlit telah di-deploy dan dapat diakses melalui tautan berikut:
[Dropout Analytics Streamlit App](https://final-project2-2pxcjgu42iernasuyde7dg.streamlit.app/)

---

Dengan mengikuti langkah-langkah ini, Anda dapat membuat dan menjalankan sistem prediksi Machine Learning untuk memprediksi kemungkinan dropout siswa berdasarkan data demografis dan akademik.

### **Conclusion**
Berdasarkan analisis data yang telah dilakukan, beberapa faktor utama yang mempengaruhi **tingkat dropout siswa** di Jaya Jaya Institut dapat diidentifikasi. Faktor-faktor ini mencakup:

1. **Usia**: Siswa yang lebih muda, terutama di kelompok usia 18-19 tahun, cenderung memiliki tingkat dropout yang lebih tinggi. Hal ini mungkin terkait dengan ketidaksiapan menghadapi transisi dari pendidikan menengah ke perguruan tinggi.
2. **Nilai Akademik**: Siswa dengan nilai akademik rendah, terutama di semester pertama, menunjukkan kecenderungan yang lebih tinggi untuk meninggalkan pendidikan mereka. Nilai ini menjadi indikator yang signifikan untuk mendeteksi potensi masalah akademik yang dapat mengarah pada keputusan dropout.
3. **Jenis Kelamin**: Terdapat perbedaan kecil dalam tingkat dropout antara siswa laki-laki dan perempuan, dengan siswa perempuan sedikit lebih cenderung untuk dropout.
4. **Pemberian Beasiswa**: Siswa yang menerima beasiswa memiliki tingkat dropout yang lebih rendah dibandingkan mereka yang tidak mendapat beasiswa, menandakan bahwa dukungan finansial dapat berperan besar dalam mempertahankan siswa.
5. **Status Keuangan**: Siswa yang tercatat memiliki utang juga menunjukkan kecenderungan lebih tinggi untuk dropout, kemungkinan besar terkait dengan masalah keuangan yang memengaruhi kelangsungan studi mereka.

Untuk memprediksi potensi dropout dengan lebih akurat, sistem prediksi berbasis **Machine Learning** dikembangkan menggunakan algoritma **Random Forest**. Model ini dilatih dengan data siswa yang mencakup faktor demografis dan akademik. Setelah evaluasi, model ini menunjukkan hasil yang baik dengan akurasi yang tinggi dan skor F1 yang memadai, yang mengindikasikan kemampuannya dalam menangani masalah ketidakseimbangan kelas.

Selain itu, dashboard interaktif yang dikembangkan menggunakan **Looker Studio** memberikan visualisasi yang sangat membantu untuk memantau dan menganalisis data dropout. Dashboard ini memungkinkan pihak manajemen untuk melihat **persentase siswa yang dropout**, **distribusi status siswa berdasarkan usia dan jenis kelamin**, serta faktor-faktor lain yang dapat memengaruhi keputusan dropout. Hal ini memberikan wawasan yang lebih dalam mengenai pola yang terjadi di Jaya Jaya Institut.

Dengan menggunakan model Machine Learning ini bersama dengan dashboard, Jaya Jaya Institut dapat lebih cepat mendeteksi siswa yang berisiko tinggi untuk dropout dan memberikan dukungan yang lebih tepat waktu dan terarah. Ini akan berkontribusi pada peningkatan tingkat kelulusan dan pengurangan tingkat dropout, yang pada gilirannya akan menjaga reputasi institusi dan meningkatkan pengalaman pendidikan bagi para siswa.


### **Rekomendasi Action Items**

1. **Intervensi Dini Berdasarkan Prediksi Dropout**  
   Dengan menggunakan model prediksi Machine Learning yang telah dikembangkan, pihak manajemen dapat mengidentifikasi siswa yang berisiko tinggi untuk dropout lebih awal. Rekomendasikan untuk melakukan **pendampingan akademik** dan **mentoring pribadi** untuk siswa yang teridentifikasi dalam kategori berisiko. Program ini bisa berupa sesi bimbingan dengan dosen atau konselor akademik untuk membantu siswa mengatasi kesulitan yang mereka hadapi.

2. **Meningkatkan Dukungan Keuangan untuk Siswa Berisiko**  
   Berdasarkan data yang menunjukkan bahwa siswa yang memiliki utang atau tidak mendapatkan beasiswa cenderung lebih sering dropout, institusi perlu mengeksplorasi **peningkatan program beasiswa** atau **program bantuan keuangan** untuk siswa yang membutuhkan. Pemberian beasiswa atau bantuan keuangan tambahan akan membantu mengurangi tekanan finansial yang menjadi faktor utama dalam keputusan dropout.

3. **Pendekatan Personalisasi Berdasarkan Usia dan Status Akademik**  
   Mengingat faktor usia dan nilai akademik berperan penting dalam keputusan dropout, institusi sebaiknya memberikan pendekatan yang lebih **personalisasi** untuk siswa di usia muda (18-19 tahun) dan siswa dengan nilai akademik rendah. Program **pendampingan akademik** yang intensif dan **workshop keterampilan belajar** dapat membantu mereka menyesuaikan diri dengan tuntutan akademik di perguruan tinggi.

4. **Program Retensi untuk Siswa Non-Beasiswa**  
   Sebagai upaya untuk menurunkan tingkat dropout di antara siswa yang tidak mendapat beasiswa, institusi dapat mengembangkan **program retensi** yang dirancang untuk meningkatkan keterlibatan siswa, seperti kegiatan ekstrakurikuler yang dapat memperkuat rasa komunitas dan mengurangi perasaan keterasingan. Ini juga bisa mencakup kegiatan sosial atau acara yang mendorong siswa untuk lebih terlibat dalam kehidupan kampus.

5. **Optimalisasi Penggunaan Dashboard untuk Monitoring**  
   Manfaatkan **dashboard interaktif** yang telah dibuat untuk memantau secara real-time tingkat dropout dan faktor-faktor yang mempengaruhinya. Dengan menggunakan data yang ada, pihak manajemen dapat secara rutin mengevaluasi dan mengambil tindakan yang diperlukan untuk menurunkan tingkat dropout. Disarankan agar **pihak administrasi** dan **staff akademik** diberikan pelatihan untuk membaca dan menginterpretasikan data ini dengan lebih efektif.

6. **Evaluasi Berkala Program Intervensi**  
   Program-program yang telah diterapkan untuk menurunkan tingkat dropout perlu dievaluasi secara berkala untuk mengukur keberhasilan dan memperbaiki pendekatan yang ada. Data yang dikumpulkan dari dashboard dan model Machine Learning harus digunakan untuk menilai efektivitas tindakan yang diambil dan melakukan penyesuaian jika diperlukan.

