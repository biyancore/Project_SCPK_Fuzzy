
# =========================================================
# IMPORT LIBRARY
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="SPK Fuzzy Mamdani",
    layout="wide",
    page_icon="🚑"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚑 SPK Fuzzy Mamdani")

menu = st.sidebar.radio(
    "Navigasi",
    [
        "🏠 Home",
        "📊 Dataset",
        "🧠 Perhitungan SPK",
        "📈 Visualisasi",
        "👥 Profil Kelompok"
    ]
)

st.sidebar.divider()

st.sidebar.write("### 👥 Anggota Kelompok")
st.sidebar.write("Briliant Priscilla (123240068)")
st.sidebar.write("Andini Papa Sulima (123240118)")

# =========================================================
# UPLOAD DATASET
# =========================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df_raw = pd.read_csv(uploaded_file) 

    df_raw.columns = df_raw.columns.str.strip()

    if len(df_raw) < 250:

        st.error(
            "❌ Dataset minimal 250 baris"
        )

        st.stop()

else:

    df_raw = None

# =========================================================
# INISIALISASI FUZZY
# =========================================================

def init_fuzzy(): #fungsi untuk inisialisasi sistem fuzzy, akan mengembalikan objek sistem fuzzy dan variabel-variabelnya

    # =====================================================
    # INPUT
    # =====================================================

    waktu = ctrl.Antecedent(
        np.arange(0, 151, 1),
        'C1'
    )

    biaya = ctrl.Antecedent(
        np.arange(0, 2001, 1),
        'C2'
    )

    cuaca = ctrl.Antecedent(
        np.arange(1, 6, 1),
        'C3'
    )

    kapasitas = ctrl.Antecedent(
        np.arange(1, 6, 1),
        'C4'
    )

    medan = ctrl.Antecedent(
        np.arange(1, 6, 1),
        'C5'
    )

    # =====================================================
    # OUTPUT
    # =====================================================

    skor = ctrl.Consequent(
        np.arange(0, 101, 1),
        'Membership'
    )

    # =====================================================
    # MEMBERSHIP FUNCTION
    # =====================================================

    waktu['cepat'] = fuzz.trimf(
        waktu.universe,
        [0, 0, 60] #mulai dari 0, puncak di 0, turun sampai 60
    )

    waktu['sedang'] = fuzz.trapmf( 
        waktu.universe,
        [40, 60, 90, 110] #mulai dari 40, puncak di 60 dan 90, turun sampai 110
    )

    waktu['lambat'] = fuzz.trimf(
        waktu.universe,
        [90, 150, 150] #mulai dari 90, puncak di 150, turun sampai 150
    )

    biaya['murah'] = fuzz.trimf(
        biaya.universe,
        [0, 0, 800] #mulai dari 0, puncak di 0, turun sampai 800
    )

    biaya['standar'] = fuzz.trimf(
        biaya.universe,
        [500, 1000, 1500] #mulai dari 500, puncak di 1000, turun sampai 1500
    )

    biaya['mahal'] = fuzz.trimf(
        biaya.universe,
        [1200, 2000, 2000] #mulai dari 1200, puncak di 2000, turun sampai 2000
    )

    for k in [cuaca, kapasitas, medan]: #untuk setiap variabel cuaca, kapasitas, medan, buat fungsi keanggotaan untuk buruk, cukup, bagus

        k['buruk'] = fuzz.trimf(
            k.universe,
            [1, 1, 3] #mulai dari 1, puncak di 1, turun sampai 3
        )

        k['cukup'] = fuzz.trimf(
            k.universe,
            [2, 3, 4] #mulai dari 2, puncak di 3, turun sampai 4        
        )

        k['bagus'] = fuzz.trimf(
            k.universe,
            [3, 5, 5] #mulai dari 3, puncak di 5, turun sampai 5
        )

    skor['sedikit'] = fuzz.trimf(
        skor.universe,
        [0, 0, 45] #mulai dari 0, puncak di 0, turun sampai 45
    )

    skor['sedang'] = fuzz.trimf(
        skor.universe,
        [35, 55, 75] #mulai dari 35, puncak di 55, turun sampai 75
    )

    skor['banyak'] = fuzz.trimf(
        skor.universe,
        [60, 100, 100] #mulai dari 60, puncak di 100, turun sampai 100
    )

    # =====================================================
    # RULES
    # =====================================================

    rules = [

        ctrl.Rule(
            waktu['cepat']
            & biaya['murah'],
            skor['banyak']
        ),

        ctrl.Rule(
            waktu['cepat']
            & cuaca['bagus'],
            skor['banyak']
        ),

        ctrl.Rule(
            waktu['cepat']
            & medan['bagus'],
            skor['banyak']
        ),

        ctrl.Rule(
            kapasitas['bagus']
            & cuaca['bagus'],
            skor['banyak']
        ),

        ctrl.Rule(
            biaya['mahal']
            | waktu['lambat'],
            skor['sedikit']
        ),

        ctrl.Rule(
            medan['buruk']
            & cuaca['buruk'],
            skor['sedikit']
        ),

        ctrl.Rule(
            kapasitas['buruk']
            & biaya['mahal'],
            skor['sedikit']
        ),

        ctrl.Rule(
            waktu['sedang']
            & biaya['standar'],
            skor['sedang']
        ),

        ctrl.Rule(
            cuaca['cukup']
            & medan['cukup'],
            skor['sedang']
        ),

        ctrl.Rule(
            kapasitas['cukup'],
            skor['sedang']
        )
    ]

    system = ctrl.ControlSystem(rules)

    return (
        system,
        waktu,
        biaya,
        cuaca,
        kapasitas,
        medan,
        skor
    )

# =========================================================
# INISIALISASI
# =========================================================

(
    fuzzy_system,
    v_waktu,
    v_biaya,
    v_cuaca,
    v_kapasitas,
    v_medan,
    v_skor
) = init_fuzzy()

# =========================================================
# HOME
# =========================================================

if menu == "🏠 Home":

    st.title("🚑 Sistem Pendukung Keputusan")

    st.subheader(
        "Metode Fuzzy Mamdani"
    )

    st.write("""
    Sistem ini digunakan untuk menentukan
    moda transportasi terbaik untuk distribusi
    obat darurat berdasarkan:

    • Waktu Distribusi

    • Biaya Distribusi

    • Kondisi Cuaca

    • Kapasitas Kendaraan

    • Kondisi Medan
    """)

# =========================================================
# DATASET
# =========================================================

elif menu == "📊 Dataset":

    st.title("📊 Dataset")

    if df_raw is not None:

        st.success(
            f"✅ Dataset berhasil dimuat ({len(df_raw)} baris)"
        )

        st.dataframe( 
            df_raw,
            use_container_width=True #agar tabel memenuhi lebar kontainer
        )

        # =================================================
        # STATISTIK
        # =================================================

        st.write("## 📈 Statistik Dataset")

        st.dataframe(
            df_raw.describe(), #menampilkan statistik deskriptif dari dataset. fungsi describe() memberikan ringkasan statistik (mean, std, min, 25%, 50%, 75%, max) untuk setiap kolom numerik dalam dataset.
            use_container_width=True
        )

    else:

        st.warning(
            "⚠️ Upload dataset terlebih dahulu"
        )

# =========================================================
# PERHITUNGAN SPK
# =========================================================

elif menu == "🧠 Perhitungan SPK":

    st.title("🧠 Perhitungan SPK")

    if df_raw is None:

        st.warning(
            "⚠️ Upload dataset terlebih dahulu"
        )

        st.stop()

    # =====================================================
    # INPUT PARAMETER
    # =====================================================

    st.write("## 🎛️ Input Parameter")

    st.info("""
    Centang kriteria yang ingin diprioritaskan
    dalam proses pengambilan keputusan.
    """)

    col1, col2 = st.columns(2) #penataan lebi rapi

    with col1:

        prioritas_c1 = st.checkbox(
            "Prioritaskan C1 - Waktu"
        )

        val_c1 = st.slider(
            "C1 - Waktu Distribusi (menit)",
            0, #mulai dari 0 menit
            150, #maksimal 150 menit
            45 #nilai default 45 menit, bisa disesuaikan oleh pengguna
        )

        prioritas_c2 = st.checkbox(
            "Prioritaskan C2 - Biaya"
        )

        val_c2 = st.slider(
            "C2 - Biaya Distribusi (rupiah)",
            0,
            2000,
            600
        )

        prioritas_c3 = st.checkbox(
            "Prioritaskan C3 - Cuaca"
        )

        val_c3 = st.slider(
            "C3 - Kondisi Cuaca",
            1,
            5,
            4,
            help="""
            1 = Sangat Buruk
            5 = Sangat Baik
            """ #tanda tanya untuk memberikan penjelasan pada slider cuaca
        )

    with col2:

        prioritas_c4 = st.checkbox(
            "Prioritaskan C4 - Kapasitas"
        )

        val_c4 = st.slider(
            "C4 - Kapasitas Kendaraan",
            1,
            5,
            3,
            help="""
            1 = Sangat Kecil
            5 = Sangat Besar
            """
        )

        prioritas_c5 = st.checkbox(
            "Prioritaskan C5 - Medan"
        )

        val_c5 = st.slider(
            "C5 - Kondisi Medan",
            1,
            5,
            4,
            help="""
            1 = Sangat Sulit
            5 = Sangat Mudah
            """
        )

    # =====================================================
    # BOBOT PRIORITAS
    # =====================================================

    bobot_c1 = 2 if prioritas_c1 else 1 #jika prioritas_c1 dicentang, bobot_c1 = 2, jika tidak dicentang, bobot_c1 = 1
    bobot_c2 = 2 if prioritas_c2 else 1 #jika prioritas_c2 dicentang, bobot_c2 = 2, jika tidak dicentang, bobot_c2 = 1
    bobot_c3 = 2 if prioritas_c3 else 1 #jika prioritas_c3 dicentang, bobot_c3 = 2, jika tidak dicentang, bobot_c3 = 1
    bobot_c4 = 2 if prioritas_c4 else 1 #jika prioritas_c4 dicentang, bobot_c4 = 2, jika tidak dicentang, bobot_c4 = 1
    bobot_c5 = 2 if prioritas_c5 else 1 #jika prioritas_c5 dicentang, bobot_c5 = 2, jika tidak dicentang, bobot_c5 = 1

    prioritas_aktif = [] #list untuk menyimpan nama kriteria yang diprioritaskan

    if prioritas_c1:
        prioritas_aktif.append("Waktu") #jika prioritas_c1 dicentang, tambahkan "Waktu" ke dalam list prioritas_aktif

    if prioritas_c2:
        prioritas_aktif.append("Biaya") #jika prioritas_c2 dicentang, tambahkan "Biaya" ke dalam list prioritas_aktif

    if prioritas_c3:
        prioritas_aktif.append("Cuaca") #jika prioritas_c3 dicentang, tambahkan "Cuaca" ke dalam list prioritas_aktif

    if prioritas_c4:
        prioritas_aktif.append("Kapasitas") #jika prioritas_c4 dicentang, tambahkan "Kapasitas" ke dalam list prioritas_aktif

    if prioritas_c5:
        prioritas_aktif.append("Medan") #jika prioritas_c5 dicentang, tambahkan "Medan" ke dalam list prioritas_aktif

    if len(prioritas_aktif) > 0: 

        st.info(
            f"""
            🎯 Sistem memprioritaskan:

            {', '.join(prioritas_aktif)}
            """
        )

    else:

        st.info(
            """
            🎯 Sistem menggunakan semua kriteria
            dengan bobot normal.
            """
        )

    # =====================================================
    # BUTTON
    # =====================================================

    btn = st.button(
        "🚀 Jalankan Perhitungan",
        type="primary"
    )

    if btn:

        hasil = []

        for i, row in df_raw.iterrows(): #untuk setiap baris dalam dataset, lakukan perhitungan fuzzy dan simpan hasilnya dalam list hasil. iterrows() digunakan untuk iterasi baris.

            sim = ctrl.ControlSystemSimulation(
                fuzzy_system
            ) #membuat objek simulasi untuk sistem fuzzy yang telah dibuat sebelumnya

            sim.input['C1'] = row['C1_Waktu']
            sim.input['C2'] = row['C2_Biaya']
            sim.input['C3'] = row['C3_Cuaca']
            sim.input['C4'] = row['C4_Kapasitas']
            sim.input['C5'] = row['C5_Medan']

            try:

                sim.compute() #melakukan perhitungan fuzzy berdasarkan input yang telah diberikan. compute() akan memproses aturan-aturan fuzzy dan menghasilkan output.

                score = sim.output['Membership'] #mengambil nilai output dari variabel 'Membership' yang telah didefinisikan sebagai output dalam sistem fuzzy. nilai ini merupakan skor fuzzy yang dihitung berdasarkan aturan-aturan yang telah dibuat.

            except:

                score = 0 #jika terjadi error saat perhitungan fuzzy (misalnya karena input yang tidak valid), maka skor akan diatur menjadi 0 sebagai nilai default.

            hasil.append(score) #menambahkan skor fuzzy yang dihitung ke dalam list hasil untuk setiap baris dalam dataset. setelah iterasi selesai, list hasil akan berisi skor fuzzy untuk setiap alternatif dalam dataset.

        # =================================================
        # DATAFRAME HASIL
        # =================================================

        df_hasil = df_raw.copy() #membuat salinan dari dataset asli untuk menyimpan hasil perhitungan fuzzy. ini dilakukan agar dataset asli tetap utuh dan hasil perhitungan dapat disimpan dalam dataframe baru tanpa mengubah data asli.

        df_hasil['Skor_Fuzzy'] = hasil #menambahkan kolom baru 'Skor_Fuzzy' ke dalam dataframe hasil untuk menyimpan skor fuzzy yang telah dihitung (berasal dari list hasil yg sudah di iterasi sebelumnya) 

        # =================================================
        # JARAK PRIORITAS
        # =================================================

        jarak = np.sqrt(

            bobot_c1 * (
                (df_hasil['C1_Waktu'] - val_c1) ** 2 #val_c1 adalah nilai input yang diberikan oleh pengguna untuk kriteria C1_Waktu (misalnya user memilih 45 menit). df_hasil['C1_Waktu'] adalah nilai kriteria C1_Waktu untuk setiap alternatif dalam dataset.
                                                     #rumus ini menghitung selisih antara nilai kriteria C1_Waktu pada setiap alternatif dengan nilai input val_c1 yang diberikan oleh pengguna, kemudian mengkuadratkan selisih tersebut dan mengalikan dengan bobot prioritas bobot_c1. 
                                                     # hasilnya akan memberikan kontribusi terhadap jarak total untuk kriteria C1_Waktu.
            ) #menghitung selisih antara nilai kriteria C1_Waktu pada setiap alternatif dengan nilai input val_c1 yang diberikan oleh pengguna, kemudian mengkuadratkan selisih tersebut dan mengalikan dengan bobot prioritas bobot_c1. hasilnya akan memberikan kontribusi terhadap jarak total untuk kriteria C1_Waktu.

            +

            bobot_c2 * (
                (df_hasil['C2_Biaya'] - val_c2) ** 2
            ) #menghitung selisih antara nilai kriteria C2_Biaya pada setiap alternatif dengan nilai input val_c2 yang diberikan oleh pengguna, kemudian mengkuadratkan selisih tersebut dan mengalikan dengan bobot prioritas bobot_c2. hasilnya akan memberikan kontribusi terhadap jarak total untuk kriteria C2_Biaya.

            +

            bobot_c3 * (
                (df_hasil['C3_Cuaca'] - val_c3) ** 2
            ) #menghitung selisih antara nilai kriteria C3_Cuaca pada setiap alternatif dengan nilai input val_c3 yang diberikan oleh pengguna, kemudian mengkuadratkan selisih tersebut dan mengalikan dengan bobot prioritas bobot_c3. hasilnya akan memberikan kontribusi terhadap jarak total untuk kriteria C3_Cuaca.

            +

            bobot_c4 * (
                (df_hasil['C4_Kapasitas'] - val_c4) ** 2
            ) #menghitung selisih antara nilai kriteria C4_Kapasitas pada setiap alternatif dengan nilai input val_c4 yang diberikan oleh pengguna, kemudian mengkuadratkan selisih tersebut dan mengalikan dengan bobot prioritas bobot_c4. hasilnya akan memberikan kontribusi terhadap jarak total untuk kriteria C4_Kapasitas.

            +

            bobot_c5 * (
                (df_hasil['C5_Medan'] - val_c5) ** 2
            ) #menghitung selisih antara nilai kriteria C5_Medan pada setiap alternatif dengan nilai input val_c5 yang diberikan oleh pengguna, kemudian mengkuadratkan selisih tersebut dan mengalikan dengan bobot prioritas bobot_c5. hasilnya akan memberikan kontribusi terhadap jarak total untuk kriteria C5_Medan.
        ) #menghitung jarak Euclidean antara nilai kriteria setiap alternatif dengan nilai input yang diberikan oleh pengguna, dengan memperhitungkan bobot prioritas untuk setiap kriteria. hasilnya disimpan dalam variabel jarak.

        max_jarak = jarak.max()

        df_hasil['Persentase_Kecocokan'] = np.clip(

            100 * (
                1 - (jarak / max_jarak)
            ),

            0,
            100 #menghitung persentase kecocokan untuk setiap alternatif berdasarkan jarak yang telah dihitung sebelumnya. rumusnya adalah 100 x (1 - (jarak / max_jarak)). Selisih tadi dijadiin persentase. 
            #hasilnya kemudian dibatasi antara 0 dan 100 menggunakan fungsi np.clip() untuk memastikan bahwa persentase kecocokan tidak melebihi 100% atau kurang dari 0%. 
            # hasil persentase kecocokan disimpan dalam kolom 'Persentase_Kecocokan' pada dataframe hasil. 
            # Hasil menyocokan dengan input pengguna, semakin kecil jarak, semakin tinggi persentase kecocokan.
            # jarak dihitung berdasarkan selisih antara nilai kriteria alternatif dengan nilai input pengguna, dengan memperhitungkan bobot prioritas. 
            # Persentase kecocokan memberikan gambaran seberapa baik alternatif tersebut sesuai dengan preferensi pengguna.
            # jarak dihitung menggunakan rumus Euclidean yang memperhitungkan bobot prioritas untuk setiap kriteria.
            #presentase kecocokan tuh cocok sama input pengguna, semakin kecil jarak, semakin tinggi persentase kecocokan. jadi kalau alternatif punya nilai kriteria yang dekat dengan input pengguna, maka persentase kecocokannya akan tinggi.
        )

        # =================================================
        # SKOR AKHIR
        # =================================================

        df_hasil['Skor_Akhir'] = (

            (0.7 * df_hasil['Persentase_Kecocokan'])

            +

            (0.3 * df_hasil['Skor_Fuzzy'])

        ) 
        #menghitung skor akhir untuk setiap alternatif dengan menggabungkan persentase kecocokan dan skor fuzzy. 
        # rumusnya adalah (0.7 x Persentase_Kecocokan) + (0.3 x Skor_Fuzzy).
        # kenapa 0.7 dan 0.3? karena kita ingin memberikan bobot lebih besar pada persentase kecocokan (70%) dibandingkan dengan skor fuzzy (30%) dalam menentukan skor akhir.

        # =================================================
        # GROUP BY MODA
        # =================================================

        df_hasil = df_hasil.loc[
            df_hasil.groupby(
                'Moda_Transportasi'
            )['Skor_Akhir'].idxmax()
        ] 
        # mengelompokkan dataframe hasil berdasarkan kolom 'Moda_Transportasi' dan mengambil baris dengan skor akhir tertinggi untuk setiap moda transportasi.

        # =================================================
        # SORTING
        # =================================================

        df_hasil = df_hasil.sort_values(
            by='Skor_Akhir',
            ascending=False
        )

        df_hasil.reset_index(
            drop=True,
            inplace=True
        ) #mengatur ulang indeks dataframe hasil setelah melakukan sorting berdasarkan skor akhir. drop=True digunakan untuk menghapus indeks lama, inplace=True digunakan untuk melakukan perubahan langsung pada dataframe hasil tanpa perlu membuat salinan baru.

        df_hasil.index += 1 #supaya ranking mulai dari 1, bukan 0

        df_hasil['Ranking'] = df_hasil.index #kolom baru 'rangking' yang berisi nilai indeks yang sudah dimulai dari 1

        # =================================================
        # BINTANG
        # =================================================

        def stars(score):

            if score >= 90:
                return "⭐⭐⭐⭐⭐"

            elif score >= 80:
                return "⭐⭐⭐⭐"

            elif score >= 70:
                return "⭐⭐⭐"

            elif score >= 60:
                return "⭐⭐"

            else:
                return "⭐"

        def status(score):

            if score >= 90:
                return "Sangat Direkomendasikan"

            elif score >= 80:
                return "Direkomendasikan"

            elif score >= 70:
                return "Cukup Direkomendasikan"

            elif score >= 60:
                return "Kurang Direkomendasikan"

            else:
                return "Tidak Direkomendasikan"

        df_hasil['Bintang'] = df_hasil[
            'Skor_Akhir'
        ].apply(stars)

        df_hasil['Keterangan'] = df_hasil[
            'Skor_Akhir'
        ].apply(status)

        st.session_state['df_hasil'] = df_hasil
        st.session_state['user_inputs'] = [val_c1, val_c2, val_c3, val_c4, val_c5]
        
        st.success("✅ Perhitungan Selesai! Silakan cek hasil di bawah atau pindah ke menu Visualisasi.")

        # =================================================
        # TABEL
        # =================================================

        st.write("## 🏆 Ranking")

        st.dataframe(

            df_hasil[
                [
                    'Ranking',
                    'ID_Alternatif',
                    'Moda_Transportasi',

                    'Persentase_Kecocokan',
                    'Skor_Fuzzy',
                    'Skor_Akhir',

                    'Bintang',
                    'Keterangan',

                    'C1_Waktu',
                    'C2_Biaya',
                    'C3_Cuaca',
                    'C4_Kapasitas',
                    'C5_Medan'
                ]
            ].head(10).style.background_gradient(

                subset=['Skor_Akhir'],
                cmap='Greens',
                vmin=70, #nilai minimum untuk warna hijau mulai dari 70, semakin tinggi skor akhir, semakin hijau warnanya. 
                vmax=100
            ),

            use_container_width=True
        )

        # =================================================
        # TOP 1
        # =================================================

        top1 = df_hasil.iloc[0]

        st.success(
            f"""
🥇 Rekomendasi Terbaik:
{top1['Moda_Transportasi']} {top1['Bintang']}

Moda transportasi ini dipilih karena memiliki:

• Skor akhir tertinggi sebesar {top1['Skor_Akhir']:.2f}%

• Tingkat kecocokan sebesar {top1['Persentase_Kecocokan']:.2f}%

• Waktu distribusi efisien ({top1['C1_Waktu']} menit)

• Biaya distribusi optimal ({top1['C2_Biaya']})

• Kapasitas dan kondisi medan mendukung

• Hasil evaluasi fuzzy terbaik
"""
        )

        # =================================================
        # BAR CHART
        # =================================================

        st.write("## 📊 Grafik Ranking")

        fig1, ax1 = plt.subplots(
            figsize=(12, 6) # ukuran grafik 12x6 inci
        )

        top5 = df_hasil.head(5) #mengambil 5 baris teratas dari dataframe hasil untuk ditampilkan dalam grafik ranking. 

        x_pos = np.arange(len(top5)) #membuat array posisi x untuk setiap bar dalam grafik berdasarkan jumlah bar yang akan ditampilkan (dalam hal ini 5). 

        bar_colors = [

            '#FFD700',
            '#008080',
            '#008080',
            '#008080',
            '#008080'
        ]

        bars = ax1.bar(

            x_pos,
            top5['Skor_Akhir'],

            width=0.55,

            color=bar_colors[:len(top5)],

            edgecolor='black',

            linewidth=1.2
        ) #membuat grafik batang (bar chart) dengan posisi x yang telah ditentukan, tinggi batang berdasarkan nilai 'Skor_Akhir' dari top5, lebar batang 0.55, warna batang sesuai dengan bar_colors, dan memberikan garis tepi hitam dengan ketebalan 1.2.

        ax1.set_xticks(x_pos) #mengatur posisi x pada sumbu x sesuai dengan array x_pos yang telah dibuat sebelumnya. ini memastikan bahwa label pada sumbu x akan ditempatkan di posisi yang benar di bawah setiap batang dalam grafik.

        ax1.set_xticklabels(
            top5['Moda_Transportasi'],
            rotation=10
        ) #mengatur label pada sumbu x dengan menggunakan nama moda transportasi dari top5. rotation=10 digunakan untuk memutar label sebesar 10 derajat agar lebih mudah dibaca jika nama moda transportasi cukup panjang.

        ax1.set_title(
            "Top 5 Moda Transportasi"
        )

        ax1.set_xlabel(
            "Moda Transportasi"
        )

        ax1.set_ylabel(
            "Skor Akhir (%)"
        )

        ax1.grid(
            axis='y',
            alpha=0.3
        ) #menambahkan grid pada sumbu y dengan tingkat transparansi 0.3 untuk membantu pembacaan nilai pada grafik.

        for bar in bars:

            yval = bar.get_height() #mengambil tinggi dari setiap batang dalam grafik, yang mewakili nilai 'Skor_Akhir' untuk setiap moda transportasi. nilai ini akan digunakan untuk menampilkan label nilai di atas setiap batang.

            ax1.text(

                bar.get_x() + bar.get_width()/2,

                yval + 2, #menempatkan teks sedikit di atas batang (dikasih jarak)

                f'{yval:.1f}%',

                ha='center'
            ) #menambahkan teks di atas setiap batang untuk menampilkan nilai 'Skor_Akhir' dalam format persentase dengan satu angka desimal. 

        st.pyplot(fig1) 

        plt.close(fig1) 

        # =================================================
        # HISTOGRAM
        # =================================================

        st.write("## 📈 Histogram Distribusi Skor") 
        #histogram digunakan untuk menunjukkan distribusi skor akhir dari seluruh alternatif moda transportasi. 
        # ini membantu kita melihat sebaran skor dan apakah ada konsentrasi skor tertentu (misalnya banyak alternatif yang memiliki skor tinggi atau rendah).
        # cara bacanya, semakin tinggi batang pada rentang skor tertentu, semakin banyak alternatif yang memiliki skor akhir dalam rentang tersebut.

        fig_hist, ax_hist = plt.subplots(
            figsize=(10, 5)
        )

        ax_hist.hist(
            df_hasil['Skor_Akhir'],
            bins=10,#jumlah batang dalam histogram, semakin banyak bins, semakin detail distribusi yang ditampilkan. 
            color='#FFA726',
            edgecolor='black'
        )

        mean_score = df_hasil[
            'Skor_Akhir'
        ].mean()

        ax_hist.axvline(
            mean_score,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean = {mean_score:.2f}'
        )

        ax_hist.legend() #untuk menampilkan legenda yang menunjukkan nilai mean pada histogram (keterangan garis vertikal merah)

        ax_hist.set_title(
            "Distribusi Skor Akhir"
        )

        ax_hist.set_xlabel(
            "Skor Akhir"
        )

        ax_hist.set_ylabel(
            "Frekuensi"
        )

        st.pyplot(fig_hist)

        st.caption("""
        Histogram menunjukkan distribusi
        skor akhir dari seluruh alternatif
        transportasi.
        """)

        plt.close(fig_hist) #menutup objek figure setelah ditampilkan untuk menghemat memori dan mencegah tumpang tindih grafik jika ada grafik lain yang akan dibuat setelahnya.

        # =================================================
        # PIE CHART
        # =================================================

        st.write("## 🥧 Distribusi Status")

        fig_pie, ax_pie = plt.subplots(
            figsize=(7, 7)
        )

        status_count = df_hasil[
            'Keterangan'
        ].value_counts() #menghitung jumlah alternatif untuk setiap kategori dalam kolom 'Keterangan' pada dataframe hasil. value_counts() akan memberikan jumlah kemunculan setiap kategori (misalnya "Sangat Direkomendasikan", "Direkomendasikan", dll.) yang kemudian akan digunakan untuk membuat pie chart distribusi status rekomendasi.

        colors = [

            '#2ECC71', #sangat direkomendasikan
            '#F1C40F', #direkomendasikan
            '#E67E22', #tidak direkomendasikan
            '#E74C3C'  #sangat tidak direkomendasikan
                #semuanya muncul otomatis urut sesuai dengan jumlah kategori yang ada di status_count
        ] 

        ax_pie.pie(

            status_count,

            labels=status_count.index,

            autopct='%1.1f%%', #menampilkan persentase pada setiap irisan pie chart dengan format 1 angka desimal

            startangle=90, #memulai irisan pertama pada sudut 90 derajat (dari atas ke bawah)

            explode=[0.03] * len(status_count), #jarak antara irisan pie chart. irisan akan sedikit terpisah dari pusat.

            colors=colors[:len(status_count)],

            shadow=True
        )

        ax_pie.set_title(
            "Distribusi Status Rekomendasi"
        )

        st.caption("""
        Pie chart menunjukkan persentase distribusi
        status rekomendasi dari seluruh alternatif
        transportasi.
        """)

        st.pyplot(fig_pie)

        plt.close(fig_pie)

        # =================================================
        # GRAFIK KEANGGOTAAN INPUT
        # =================================================

        st.write("## 📈 Grafik Keanggotaan")

        tabs = st.tabs([
            "C1 Waktu",
            "C2 Biaya",
            "C3 Cuaca",
            "C4 Kapasitas",
            "C5 Medan"
        ])

        variabels = [

            (v_waktu, "C1 Waktu"),
            (v_biaya, "C2 Biaya"),
            (v_cuaca, "C3 Cuaca"),
            (v_kapasitas, "C4 Kapasitas"),
            (v_medan, "C5 Medan")

        ]

        input_values = [

            val_c1,
            val_c2,
            val_c3,
            val_c4,
            val_c5

        ] #list yang berisi nilai input yang diberikan oleh pengguna untuk setiap kriteria, sesuai dengan urutan variabels. nilai-nilai ini akan digunakan untuk menampilkan garis vertikal pada grafik keanggotaan yang menunjukkan posisi input pengguna dalam fungsi keanggotaan untuk setiap kriteria.

        for tab, (var, title) in zip( 
            tabs,
            variabels
        ):

            with tab:

                fig_var, ax_var = plt.subplots( #membuat objek figure dan axes untuk setiap tab yang akan menampilkan grafik keanggotaan untuk setiap kriteria. zip() digunakan untuk menggabungkan tabs dengan variabels sehingga setiap tab akan menampilkan grafik untuk variabel yang sesuai. 
                    figsize=(7, 4)
                ) 

                for term in var.terms: #

                    ax_var.plot(
                        var.universe,
                        var[term].mf,
                        linewidth=2,
                        label=term
                    )

                current_input = input_values[
                    variabels.index((var, title))
                ]

                ax_var.axvline(
                    current_input,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label='Input User'
                )

                ax_var.legend()

                ax_var.set_xlabel(
                    "Nilai Kriteria"
                )

                ax_var.set_ylabel(
                    "Derajat Keanggotaan"
                )

                ax_var.grid(alpha=0.3)

                st.pyplot(fig_var)

                plt.close(fig_var)

        # =================================================
        # MEMBERSHIP FUNCTION
        # =================================================

        st.write("## 📉 Membership Function")

        nilai_skor = top1['Skor_Akhir']

        fig4, ax4 = plt.subplots(
            figsize=(10, 5)
        )

        ax4.plot(
            v_skor.universe,
            v_skor['sedikit'].mf,
            label='sedikit'
        )

        ax4.plot(
            v_skor.universe,
            v_skor['sedang'].mf,
            label='sedang'
        )

        ax4.plot(
            v_skor.universe,
            v_skor['banyak'].mf,
            label='banyak'
        )

        tinggi_arsir = fuzz.interp_membership(
            v_skor.universe,
            v_skor['banyak'].mf,
            nilai_skor
        )

        mf_arsir = np.minimum(
            tinggi_arsir,
            v_skor['banyak'].mf
        )

        ax4.fill_between(
            v_skor.universe,
            0,
            mf_arsir,
            facecolor='#A8D5A2',
            edgecolor='green',
            alpha=0.85
        )

        ax4.vlines(
            x=nilai_skor,
            ymin=0,
            ymax=tinggi_arsir,
            color='black',
            linewidth=4
        )

        ax4.text(
            nilai_skor + 2,
            tinggi_arsir + 0.03,
            f'{nilai_skor:.2f}%'
        )

        ax4.set_xlabel(
            "Skor Kelayakan"
        )

        ax4.set_ylabel(
            "Derajat Keanggotaan"
        )

        ax4.legend()

        st.pyplot(fig4)

        plt.close(fig4)

# =========================================================
# 📈 VISUALISASI 
# =========================================================
elif menu == "📈 Visualisasi":
    st.title("📈 Visualisasi & Analisis Radar")

    # Proteksi agar tidak error jika user langsung klik menu visualisasi tanpa hitung dulu
    if 'df_hasil' not in st.session_state:
        st.warning("⚠️ Silakan jalankan perhitungan terlebih dahulu di menu 🧠 Perhitungan SPK")
        st.stop()

    # Panggil kembali data yang disimpan di session_state
    df_hasil = st.session_state['df_hasil']
    val_c1, val_c2, val_c3, val_c4, val_c5 = st.session_state['user_inputs']

    # -----------------------------------------------------
    # 🕸️ PEMBUATAN GRAFIK RADAR SINKRON
    # -----------------------------------------------------
    st.write("## 🕸️ Grafik Radar: Perbandingan Spesifikasi vs Kebutuhan User")
    st.info("Grafik ini membandingkan parameter kondisi lapangan (input slider) dengan spesifikasi kendaraan.")

    # Dropdown interaktif untuk memilih kendaraan yang mau dicek radarnya
    pilihan_moda = st.selectbox("Pilih Moda Transportasi untuk Dianalisis:", df_hasil['Moda_Transportasi'].unique())
    data_moda = df_hasil[df_hasil['Moda_Transportasi'] == pilihan_moda].iloc[0]

    # PROSES NORMALISASI AKURAT (Skala 0 s.d 1) 
    # C1 Waktu (Makin cepat makin bagus, menggunakan batas max 150)
    user_norm_c1 = (150 - val_c1) / 150
    moda_norm_c1 = (150 - data_moda['C1_Waktu']) / 150

    # C2 Biaya (Makin murah makin bagus, menggunakan batas max 2000)
    user_norm_c2 = (2000 - val_c2) / 2000
    moda_norm_c2 = (2000 - data_moda['C2_Biaya']) / 2000

    # C3, C4, C5 (Benefit criteria: Skala 1 - 5)
    user_norm_c3 = (val_c3 - 1) / 4 if val_c3 > 1 else 0
    moda_norm_c3 = (data_moda['C3_Cuaca'] - 1) / 4 if data_moda['C3_Cuaca'] > 1 else 0
    
    user_norm_c4 = (val_c4 - 1) / 4 if val_c4 > 1 else 0
    moda_norm_c4 = (data_moda['C4_Kapasitas'] - 1) / 4 if data_moda['C4_Kapasitas'] > 1 else 0
    
    user_norm_c5 = (val_c5 - 1) / 4 if val_c5 > 1 else 0
    moda_norm_c5 = (data_moda['C5_Medan'] - 1) / 4 if data_moda['C5_Medan'] > 1 else 0

    # Pengaturan sumbu dan sudut radar
    labels = ['C1 Waktu', 'C2 Biaya', 'C3 Cuaca', 'C4 Kapasitas', 'C5 Medan']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Menutup lingkaran radar

    values_user = [user_norm_c1, user_norm_c2, user_norm_c3, user_norm_c4, user_norm_c5]
    values_user += values_user[:1]

    values_moda = [moda_norm_c1, moda_norm_c2, moda_norm_c3, moda_norm_c4, moda_norm_c5]
    values_moda += values_moda[:1]

    # Plotting Radar Chart ke Streamlit
    fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Area Merah = Garis Kebutuhan dari Slider User
    ax_radar.plot(angles, values_user, color='red', linewidth=2, linestyle='--', label='Kebutuhan Lapangan (User)')
    ax_radar.fill(angles, values_user, color='red', alpha=0.1)

    # Area Hijau Toska = Spesifikasi Armada hasil Dataset
    ax_radar.plot(angles, values_moda, color='teal', linewidth=2, label=f"Spesifikasi: {pilihan_moda}")
    ax_radar.fill(angles, values_moda, color='teal', alpha=0.2)

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax_radar.set_ylim(0, 1)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    st.pyplot(fig_radar)
    plt.close(fig_radar)
        
# =========================================================
# PROFIL
# =========================================================

elif menu == "👥 Profil Kelompok":

    st.title("👥 Profil Kelompok")

    st.write("""
    ### Mata Kuliah
    Sistem Pendukung Keputusan

    ### Metode
    Fuzzy Mamdani

    ### Anggota Kelompok

    1. Briliant Priscilla

    2. Andini Papa Sulima
    """)
