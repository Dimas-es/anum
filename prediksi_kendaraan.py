import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Load data dari Excel
# Ganti baris ini agar membaca file Excel
# df = pd.read_csv('data_kendaraan.csv')
df = pd.read_excel('jmlh_kndrn_brmtr_brdskn_jns_kndrn_d_kt_tskmly.xlsx')

# 2. Pastikan nama kolom sesuai
# Jika nama kolom berbeda, sesuaikan di sini
df.columns = [col.strip().lower() for col in df.columns]

# Tampilkan nama-nama kolom untuk membantu identifikasi
print('Nama-nama kolom pada file Excel:')
print(df.columns)

# Contoh penyesuaian nama kolom
# id, de_prov, nama_prov, kabupaten, nama_kabupaten, jenis_kendaraan, jumlah_kendaraan, satuan, tahun

# 2. Filter data untuk Kota Tasikmalaya
df = df[df['nama_kabupaten_kota'].str.contains('KOTA TASIK', case=False)]

# 3. Prediksi per jenis kendaraan
hasil_prediksi = {}

for jenis in df['jenis_kendaraan'].unique():
    data_jenis = df[df['jenis_kendaraan'] == jenis]
    X = data_jenis['tahun'].values.reshape(-1, 1)
    y = data_jenis['jumlah_kendaraan'].values

    # Regresi Linier
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    tahun_prediksi = np.arange(2024, 2029).reshape(-1, 1)
    prediksi_regresi = model.predict(tahun_prediksi)

    # Interpolasi (linear)
    interpolasi = np.interp(np.arange(2024, 2029), data_jenis['tahun'], data_jenis['jumlah_kendaraan'])

    hasil_prediksi[jenis] = {
        'tahun': np.arange(2024, 2029),
        'regresi': prediksi_regresi,
        'interpolasi': interpolasi,
        'slope': slope,
        'intercept': intercept,
        'tahun_aktual': data_jenis['tahun'].values,
        'jumlah_aktual': data_jenis['jumlah_kendaraan'].values
    }

# 4. Tampilkan hasil prediksi dan perhitungannya
for jenis, hasil in hasil_prediksi.items():
    print(f"\nJenis: {jenis}")
    print("Data Aktual:")
    for t, j in zip(hasil['tahun_aktual'], hasil['jumlah_aktual']):
        print(f"  Tahun {t}: {j}")

    print("\nRegresi Linier:")
    print(f"  Persamaan: y = {hasil['slope']:.2f} * x + {hasil['intercept']:.2f}")
    print("  Koefisien (slope):", hasil['slope'])
    print("  Intercept:", hasil['intercept'])

    print("\nPrediksi Regresi Linier:")
    for t, r in zip(hasil['tahun'], hasil['regresi']):
        print(f"  Tahun {t}: {int(r)}")

    print("\nPrediksi Interpolasi Linear:")
    for t, i in zip(hasil['tahun'], hasil['interpolasi']):
        print(f"  Tahun {t}: {int(i)}")
