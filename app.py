
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

def init_fuzzy():

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
        [0, 0, 60]
    )

    waktu['sedang'] = fuzz.trapmf(
        waktu.universe,
        [40, 60, 90, 110]
    )

    waktu['lambat'] = fuzz.trimf(
        waktu.universe,
        [90, 150, 150]
    )

    biaya['murah'] = fuzz.trimf(
        biaya.universe,
        [0, 0, 800]
    )

    biaya['standar'] = fuzz.trimf(
        biaya.universe,
        [500, 1000, 1500]
    )

    biaya['mahal'] = fuzz.trimf(
        biaya.universe,
        [1200, 2000, 2000]
    )

    for k in [cuaca, kapasitas, medan]:

        k['buruk'] = fuzz.trimf(
            k.universe,
            [1, 1, 3]
        )

        k['cukup'] = fuzz.trimf(
            k.universe,
            [2, 3, 4]
        )

        k['bagus'] = fuzz.trimf(
            k.universe,
            [3, 5, 5]
        )

    skor['sedikit'] = fuzz.trimf(
        skor.universe,
        [0, 0, 45]
    )

    skor['sedang'] = fuzz.trimf(
        skor.universe,
        [35, 55, 75]
    )

    skor['banyak'] = fuzz.trimf(
        skor.universe,
        [60, 100, 100]
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
            use_container_width=True
        )

        # =================================================
        # STATISTIK
        # =================================================

        st.write("## 📈 Statistik Dataset")

        st.dataframe(
            df_raw.describe(),
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

    col1, col2 = st.columns(2)

    with col1:

        prioritas_c1 = st.checkbox(
            "Prioritaskan C1 - Waktu"
        )

        val_c1 = st.slider(
            "C1 - Waktu Distribusi (menit)",
            0,
            150,
            45
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
            """
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

    bobot_c1 = 2 if prioritas_c1 else 1
    bobot_c2 = 2 if prioritas_c2 else 1
    bobot_c3 = 2 if prioritas_c3 else 1
    bobot_c4 = 2 if prioritas_c4 else 1
    bobot_c5 = 2 if prioritas_c5 else 1

    prioritas_aktif = []

    if prioritas_c1:
        prioritas_aktif.append("Waktu")

    if prioritas_c2:
        prioritas_aktif.append("Biaya")

    if prioritas_c3:
        prioritas_aktif.append("Cuaca")

    if prioritas_c4:
        prioritas_aktif.append("Kapasitas")

    if prioritas_c5:
        prioritas_aktif.append("Medan")

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

        for i, row in df_raw.iterrows():

            sim = ctrl.ControlSystemSimulation(
                fuzzy_system
            )

            sim.input['C1'] = row['C1_Waktu']
            sim.input['C2'] = row['C2_Biaya']
            sim.input['C3'] = row['C3_Cuaca']
            sim.input['C4'] = row['C4_Kapasitas']
            sim.input['C5'] = row['C5_Medan']

            try:

                sim.compute()

                score = sim.output['Membership']

            except:

                score = 0

            hasil.append(score)

        # =================================================
        # DATAFRAME HASIL
        # =================================================

        df_hasil = df_raw.copy()

        df_hasil['Skor_Fuzzy'] = hasil

        # =================================================
        # JARAK PRIORITAS
        # =================================================

        jarak = np.sqrt(

            bobot_c1 * (
                (df_hasil['C1_Waktu'] - val_c1) ** 2
            )

            +

            bobot_c2 * (
                (df_hasil['C2_Biaya'] - val_c2) ** 2
            )

            +

            bobot_c3 * (
                (df_hasil['C3_Cuaca'] - val_c3) ** 2
            )

            +

            bobot_c4 * (
                (df_hasil['C4_Kapasitas'] - val_c4) ** 2
            )

            +

            bobot_c5 * (
                (df_hasil['C5_Medan'] - val_c5) ** 2
            )
        )

        max_jarak = jarak.max()

        df_hasil['Persentase_Kecocokan'] = np.clip(

            100 * (
                1 - (jarak / max_jarak)
            ),

            0,
            100
        )

        # =================================================
        # SKOR AKHIR
        # =================================================

        df_hasil['Skor_Akhir'] = (

            (0.7 * df_hasil['Persentase_Kecocokan'])

            +

            (0.3 * df_hasil['Skor_Fuzzy'])

        )

        # =================================================
        # GROUP BY MODA
        # =================================================

        df_hasil = df_hasil.loc[
            df_hasil.groupby(
                'Moda_Transportasi'
            )['Skor_Akhir'].idxmax()
        ]

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
        )

        df_hasil.index += 1

        df_hasil['Ranking'] = df_hasil.index

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
                vmin=70,
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
            figsize=(12, 6)
        )

        top5 = df_hasil.head(5)

        x_pos = np.arange(len(top5))

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
        )

        ax1.set_xticks(x_pos)

        ax1.set_xticklabels(
            top5['Moda_Transportasi'],
            rotation=10
        )

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
        )

        for bar in bars:

            yval = bar.get_height()

            ax1.text(

                bar.get_x() + bar.get_width()/2,

                yval + 2,

                f'{yval:.1f}%',

                ha='center'
            )

        st.pyplot(fig1)

        plt.close(fig1)

        # =================================================
        # HISTOGRAM
        # =================================================

        st.write("## 📈 Histogram Distribusi Skor")

        fig_hist, ax_hist = plt.subplots(
            figsize=(10, 5)
        )

        ax_hist.hist(
            df_hasil['Skor_Akhir'],
            bins=10,
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

        ax_hist.legend()

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

        plt.close(fig_hist)

        # =================================================
        # PIE CHART
        # =================================================

        st.write("## 🥧 Distribusi Status")

        fig_pie, ax_pie = plt.subplots(
            figsize=(7, 7)
        )

        status_count = df_hasil[
            'Keterangan'
        ].value_counts()

        colors = [

            '#2ECC71',
            '#F1C40F',
            '#E67E22',
            '#E74C3C'

        ]

        ax_pie.pie(

            status_count,

            labels=status_count.index,

            autopct='%1.1f%%',

            startangle=90,

            explode=[0.03] * len(status_count),

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

        ]

        for tab, (var, title) in zip(
            tabs,
            variabels
        ):

            with tab:

                fig_var, ax_var = plt.subplots(
                    figsize=(7, 4)
                )

                for term in var.terms:

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
# VISUALISASI
# =========================================================

elif menu == "📈 Visualisasi":

    st.title("📈 Visualisasi Dataset")

    if df_raw is not None:

        st.write(
            "## 🎯 Radar Chart Alternatif Terbaik"
        )

        # =================================================
        # HITUNG SKOR FUZZY
        # =================================================

        hasil_visual = []

        for i, row in df_raw.iterrows():

            sim = ctrl.ControlSystemSimulation(
                fuzzy_system
            )

            sim.input['C1'] = row['C1_Waktu']
            sim.input['C2'] = row['C2_Biaya']
            sim.input['C3'] = row['C3_Cuaca']
            sim.input['C4'] = row['C4_Kapasitas']
            sim.input['C5'] = row['C5_Medan']

            try:

                sim.compute()

                score = sim.output['Membership']

            except:

                score = 0

            hasil_visual.append(score)

        # =================================================
        # DATAFRAME VISUAL
        # =================================================

        df_visual = df_raw.copy()

        df_visual['Skor_Fuzzy'] = hasil_visual

        # =================================================
        # AMBIL TOP 1
        # =================================================

        top1 = df_visual.sort_values(
            by='Skor_Fuzzy',
            ascending=False
        ).iloc[0]

        # =================================================
        # DATA RADAR
        # =================================================

        categories = [

            'C1_Waktu',
            'C2_Biaya',
            'C3_Cuaca',
            'C4_Kapasitas',
            'C5_Medan'

        ]

        values = [

            top1['C1_Waktu'],
            top1['C2_Biaya'],
            top1['C3_Cuaca'],
            top1['C4_Kapasitas'],
            top1['C5_Medan']

        ]

        # =================================================
        # NORMALISASI
        # =================================================

        values_normalized = [

            values[0] / 150,
            values[1] / 2000,
            values[2] / 5,
            values[3] / 5,
            values[4] / 5

        ]

        # =================================================
        # RADAR CHART
        # =================================================

        angles = np.linspace(
            0,
            2 * np.pi,
            len(categories),
            endpoint=False
        ).tolist()

        values_normalized += values_normalized[:1]

        angles += angles[:1]

        fig, ax = plt.subplots(
            figsize=(8, 8),
            subplot_kw=dict(polar=True)
        )

        ax.plot(
            angles,
            values_normalized,
            linewidth=2
        )

        ax.fill(
            angles,
            values_normalized,
            alpha=0.25
        )

        ax.set_xticks(
            angles[:-1]
        )

        ax.set_xticklabels(
            categories
        )

        ax.set_title(
            "Profil Alternatif Terbaik",
            size=14,
            fontweight='bold'
        )

        ax.set_ylabel(
            "Nilai Normalisasi",
            labelpad=20
        )

        st.pyplot(fig)

        plt.close()

        # =================================================
        # KETERANGAN
        # =================================================

        st.info(f"""
Radar chart digunakan untuk menampilkan
profil alternatif terbaik berdasarkan
seluruh kriteria SPK.

Alternatif terbaik:
{top1['Moda_Transportasi']}

Visualisasi ini membantu melihat:

• Kriteria yang paling unggul

• Keseimbangan performa antar kriteria

• Tingkat kelayakan alternatif
dalam distribusi obat darurat
""")


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
