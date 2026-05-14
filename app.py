import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-interaktif untuk Streamlit
import matplotlib.pyplot as plt

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SPK Transportasi Darurat", layout="wide", page_icon="🚑")

# --- 2. SIDEBAR ---
st.sidebar.title("👥 Anggota Kelompok")
st.sidebar.write("**Briliant Priscilla** (123240068)")
st.sidebar.write("**Andini Papa Sulima** (123240118)")
st.sidebar.divider()

st.sidebar.header("⚙️ Parameter Keputusan")
threshold = st.sidebar.slider("Threshold Skor", 0.0, 100.0, 50.0)
st.sidebar.info("Hasil akan disaring berdasarkan skor minimal ini (0-100).")

# --- 3. LOGIKA FUZZY (SELARAS DENGAN NOTEBOOK) ---
def proses_spk_fuzzy(df_input):
    # Semesta Pembicaraan (INPUT)
    waktu = ctrl.Antecedent(np.arange(0, 151, 1), 'C1')      # Waktu 0-150 menit
    biaya = ctrl.Antecedent(np.arange(0, 2001, 1), 'C2')     # Biaya 0-2000 ribu
    cuaca = ctrl.Antecedent(np.arange(1, 6, 1), 'C3')        # Skala 1-5
    kapasitas = ctrl.Antecedent(np.arange(1, 6, 1), 'C4')    # Skala 1-5
    medan = ctrl.Antecedent(np.arange(1, 6, 1), 'C5')        # Skala 1-5
    
    # OUTPUT
    skor = ctrl.Consequent(np.arange(0, 101, 1), 'skor')    # Skor 0-100

    # --- MEMBERSHIP FUNCTIONS ---
    # C1: Waktu (Cepat, Sedang, Lambat)
    waktu['cepat'] = fuzz.trimf(waktu.universe, [0, 0, 60])
    waktu['sedang'] = fuzz.trapmf(waktu.universe, [40, 60, 90, 110])
    waktu['lambat'] = fuzz.trimf(waktu.universe, [90, 150, 150])

    # C2: Biaya (Murah, Standar, Mahal)
    biaya['murah'] = fuzz.trimf(biaya.universe, [0, 0, 800])
    biaya['standar'] = fuzz.trimf(biaya.universe, [500, 1000, 1500])
    biaya['mahal'] = fuzz.trimf(biaya.universe, [1200, 2000, 2000])

    # C3, C4, C5: (Buruk, Cukup, Bagus)
    for k in [cuaca, kapasitas, medan]:
        k['buruk'] = fuzz.trimf(k.universe, [1, 1, 3])
        k['cukup'] = fuzz.trimf(k.universe, [2, 3, 4])
        k['bagus'] = fuzz.trimf(k.universe, [3, 5, 5])

    # Output Skor (Rendah, Menengah, Tinggi)
    skor['rendah'] = fuzz.trimf(skor.universe, [0, 0, 50])
    skor['menengah'] = fuzz.trimf(skor.universe, [30, 50, 70])
    skor['tinggi'] = fuzz.trimf(skor.universe, [50, 100, 100])

    # --- RULE BASE (10 RULES) ---
    rules = [
        ctrl.Rule(waktu['cepat'] & biaya['murah'], skor['tinggi']),
        ctrl.Rule(waktu['lambat'] | biaya['mahal'], skor['rendah']),
        ctrl.Rule(cuaca['buruk'] & medan['buruk'], skor['rendah']),
        ctrl.Rule(kapasitas['bagus'] & medan['bagus'], skor['tinggi']),
        ctrl.Rule(waktu['sedang'] & biaya['standar'], skor['menengah']),
        ctrl.Rule(cuaca['bagus'] & kapasitas['bagus'], skor['tinggi']),
        ctrl.Rule(medan['cukup'] & biaya['standar'], skor['menengah']),
        ctrl.Rule(waktu['cepat'] & cuaca['cukup'], skor['tinggi']),
        ctrl.Rule(kapasitas['buruk'] | biaya['mahal'], skor['rendah']),
        ctrl.Rule(medan['bagus'] & waktu['sedang'], skor['tinggi'])
    ]

    # Masukkan semua rules ke sistem
    spk_ctrl = ctrl.ControlSystem(rules)
    simulasi = ctrl.ControlSystemSimulation(spk_ctrl)

    hasil_skor = []
    for i in range(len(df_input)):
        # INPUT 5 KRITERIA
        simulasi.input['C1'] = df_input.iloc[i]['C1_Waktu']
        simulasi.input['C2'] = df_input.iloc[i]['C2_Biaya']
        simulasi.input['C3'] = df_input.iloc[i]['C3_Cuaca']
        simulasi.input['C4'] = df_input.iloc[i]['C4_Kapasitas']
        simulasi.input['C5'] = df_input.iloc[i]['C5_Medan']
        try:
            simulasi.compute()
            res = simulasi.output['skor']
        except:
            res = 0
        hasil_skor.append(res)
    
    return hasil_skor, waktu, biaya, cuaca, kapasitas, medan

# --- 4. TAMPILAN ANTARMUKA ---
st.title("🚑 Sistem Cerdas Pendukung Keputusan")
st.subheader("Pemilihan Moda Transportasi Darurat Distribusi Obat")

try:
    # Membaca data dengan pembersihan otomatis
    df_raw = pd.read_csv("data.csv", encoding='utf-8-sig')
    df_raw.columns = df_raw.columns.str.strip()
except Exception as e:
    st.error(f"Gagal memuat file: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Dataset", "📉 Kurva Fuzzy", "🏆 Ranking"])

with tab1:
    st.write("### Data Alternatif")
    st.dataframe(df_raw, width='stretch')

with tab2:
    st.write("### Visualisasi Kurva Fuzzifikasi (5 Kriteria)")

    try:
        # Ambil objek fuzzy
        _, v_waktu, v_biaya, v_cuaca, v_kapasitas, v_medan = proses_spk_fuzzy(df_raw.head(1))
        
        # Visualisasi 3 kolom pertama
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**C1: Waktu (Menit)**")
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            for label in v_waktu.terms:
                ax1.plot(v_waktu.universe, v_waktu[label].mf, label=label, linewidth=2)
            ax1.set_title("Waktu")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig1)
            plt.close(fig1)
            
        with col2:
            st.write("**C2: Biaya (Ribu)**")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            for label in v_biaya.terms:
                ax2.plot(v_biaya.universe, v_biaya[label].mf, label=label, linewidth=2)
            ax2.set_title("Biaya")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig2)
            plt.close(fig2)
            
        with col3:
            st.write("**C3: Cuaca (Skala 1-5)**")
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            for label in v_cuaca.terms:
                ax3.plot(v_cuaca.universe, v_cuaca[label].mf, label=label, linewidth=2)
            ax3.set_title("Cuaca")
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig3)
            plt.close(fig3)
        
        # Visualisasi 2 kolom berikutnya
        col4, col5 = st.columns(2)
        
        with col4:
            st.write("**C4: Kapasitas (Skala 1-5)**")
            fig4, ax4 = plt.subplots(figsize=(5, 3))
            for label in v_kapasitas.terms:
                ax4.plot(v_kapasitas.universe, v_kapasitas[label].mf, label=label, linewidth=2)
            ax4.set_title("Kapasitas")
            ax4.legend()
            ax4.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig4)
            plt.close(fig4)
            
        with col5:
            st.write("**C5: Medan (Skala 1-5)**")
            fig5, ax5 = plt.subplots(figsize=(5, 3))
            for label in v_medan.terms:
                ax5.plot(v_medan.universe, v_medan[label].mf, label=label, linewidth=2)
            ax5.set_title("Medan")
            ax5.legend()
            ax5.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig5)
            plt.close(fig5)
            
    except Exception as e:
        st.error(f"Grafik gagal muncul: {e}")
with tab3:
    if st.button("🚀 Jalankan Perhitungan"):
        with st.spinner("Menghitung skor..."):
            # 1. Hitung skor semua data
            skor_final, _, _, _, _, _ = proses_spk_fuzzy(df_raw)
            df_raw['Skor_Mamdani'] = skor_final
            
            # 2. PROSES FILTER & SORTING
            df_sorted = df_raw[df_raw['Skor_Mamdani'] >= threshold].sort_values(by='Skor_Mamdani', ascending=False)
            
            # 3. TAMPILKAN HASILNYA
            if not df_sorted.empty:
                st.success(f"Ditemukan {len(df_sorted)} alternatif terbaik.")
                # Tampilkan tabel dengan semua kriteria
                st.dataframe(df_sorted[['ID_Alternatif', 'Moda_Transportasi', 'C1_Waktu', 'C2_Biaya', 'C3_Cuaca', 'C4_Kapasitas', 'C5_Medan', 'Skor_Mamdani']], width='stretch')
                
                st.write("### Grafik 10 Besar")
                top_10 = df_sorted.head(10)
                fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
                ax_bar.barh(top_10['Moda_Transportasi'], top_10['Skor_Mamdani'], color='teal')
                ax_bar.set_xlabel("Skor Mamdani")
                ax_bar.invert_yaxis()
                st.pyplot(fig_bar)
                plt.close(fig_bar)
            else:
                st.warning("Tidak ada yang lolos threshold.")
    else:
        st.info("Silakan klik tombol untuk memulai perhitungan.")