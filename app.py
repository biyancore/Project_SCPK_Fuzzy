import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="SPK Transportasi Darurat",
    layout="wide",
    page_icon="🚑"
)

# ============================================================
# 2. SIDEBAR — 3 WIDGET INTERAKTIF
# ============================================================
st.sidebar.title("👥 Anggota Kelompok")
st.sidebar.write("**Briliant Priscilla** (123240068)")
st.sidebar.write("**Andini Papa Sulima** (123240118)")
st.sidebar.divider()
st.sidebar.header("⚙️ Parameter Keputusan")

# Widget 1 — Threshold Skor
threshold = st.sidebar.slider("🎯 Threshold Skor Minimum", 0.0, 100.0, 50.0, step=1.0)
st.sidebar.caption("Hanya tampilkan alternatif dengan skor ≥ nilai ini.")

# Widget 2 — Filter Moda Transportasi
filter_moda = st.sidebar.selectbox(
    "🚗 Filter Moda Transportasi",
    ["Semua", "Petrol Jeep", "Motorcycle (Cargo)", "EV Delivery", "Van", "Small Truck"]
)

# Widget 3 — Jumlah Top N
top_n = st.sidebar.slider("🏆 Tampilkan Top N Alternatif", 5, 50, 10, step=5)
st.sidebar.caption("Batas jumlah baris pada grafik hasil ranking.")

# ============================================================
# 3. LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Gagal membaca data.csv: {e}")
    st.stop()

df_filtered = (
    df_raw[df_raw["Moda_Transportasi"] == filter_moda].reset_index(drop=True)
    if filter_moda != "Semua" else df_raw.copy().reset_index(drop=True)
)

# ============================================================
# 4. FUNGSI-FUNGSI FUZZY
# ============================================================
def buat_fuzzy_vars():
    """Membuat variabel fuzzy (Antecedent + Consequent + MF)."""
    waktu     = ctrl.Antecedent(np.arange(0, 151, 1), "C1")
    biaya     = ctrl.Antecedent(np.arange(0, 2001, 1), "C2")
    cuaca     = ctrl.Antecedent(np.arange(1, 6, 1),   "C3")
    kapasitas = ctrl.Antecedent(np.arange(1, 6, 1),   "C4")
    medan     = ctrl.Antecedent(np.arange(1, 6, 1),   "C5")
    skor      = ctrl.Consequent(np.arange(0, 101, 1), "skor")

    # C1: Waktu
    waktu["cepat"]  = fuzz.trimf(waktu.universe,  [0,   0,   60])
    waktu["sedang"] = fuzz.trapmf(waktu.universe, [40,  60,  90, 110])
    waktu["lambat"] = fuzz.trimf(waktu.universe,  [90, 150, 150])

    # C2: Biaya
    biaya["murah"]   = fuzz.trimf(biaya.universe, [0,    0,    800])
    biaya["standar"] = fuzz.trimf(biaya.universe, [500,  1000, 1500])
    biaya["mahal"]   = fuzz.trimf(biaya.universe, [1200, 2000, 2000])

    # C3, C4, C5: Skala 1-5
    for k in [cuaca, kapasitas, medan]:
        k["buruk"] = fuzz.trimf(k.universe, [1, 1, 3])
        k["cukup"] = fuzz.trimf(k.universe, [2, 3, 4])
        k["bagus"] = fuzz.trimf(k.universe, [3, 5, 5])

    # Output
    skor["rendah"]   = fuzz.trimf(skor.universe, [0,   0,   50])
    skor["menengah"] = fuzz.trimf(skor.universe, [30,  50,  70])
    skor["tinggi"]   = fuzz.trimf(skor.universe, [50, 100, 100])

    return waktu, biaya, cuaca, kapasitas, medan, skor


def hitung_skor_fuzzy(df_input):
    """Menjalankan inferensi Fuzzy Mamdani dan mengembalikan list skor."""
    waktu, biaya, cuaca, kapasitas, medan, skor = buat_fuzzy_vars()

    rules = [
        ctrl.Rule(waktu["cepat"]     & biaya["murah"],     skor["tinggi"]),
        ctrl.Rule(waktu["lambat"]    | biaya["mahal"],     skor["rendah"]),
        ctrl.Rule(cuaca["buruk"]     & medan["buruk"],     skor["rendah"]),
        ctrl.Rule(kapasitas["bagus"] & medan["bagus"],     skor["tinggi"]),
        ctrl.Rule(waktu["sedang"]    & biaya["standar"],   skor["menengah"]),
        ctrl.Rule(cuaca["bagus"]     & kapasitas["bagus"], skor["tinggi"]),
        ctrl.Rule(medan["cukup"]     & biaya["standar"],   skor["menengah"]),
        ctrl.Rule(waktu["cepat"]     & cuaca["cukup"],     skor["tinggi"]),
        ctrl.Rule(kapasitas["buruk"] | biaya["mahal"],     skor["rendah"]),
        ctrl.Rule(medan["bagus"]     & waktu["sedang"],    skor["tinggi"]),
    ]

    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    hasil = []
    for _, row in df_input.iterrows():
        sim.input["C1"] = row["C1_Waktu"]
        sim.input["C2"] = row["C2_Biaya"]
        sim.input["C3"] = row["C3_Cuaca"]
        sim.input["C4"] = row["C4_Kapasitas"]
        sim.input["C5"] = row["C5_Medan"]
        try:
            sim.compute()
            hasil.append(round(sim.output["skor"], 4))
        except:
            hasil.append(0.0)
    return hasil


def hitung_fuzzifikasi(df_input):
    """Menghitung derajat keanggotaan (µ) setiap nilai input terhadap himpunan fuzzy."""
    u_w = np.arange(0, 151, 1)
    u_b = np.arange(0, 2001, 1)
    u_s = np.arange(1, 6, 1)

    mf_w = {
        "Cepat":  fuzz.trimf(u_w, [0,   0,   60]),
        "Sedang": fuzz.trapmf(u_w, [40,  60,  90, 110]),
        "Lambat": fuzz.trimf(u_w, [90, 150, 150]),
    }
    mf_b = {
        "Murah":   fuzz.trimf(u_b, [0,    0,    800]),
        "Standar": fuzz.trimf(u_b, [500,  1000, 1500]),
        "Mahal":   fuzz.trimf(u_b, [1200, 2000, 2000]),
    }
    mf_s = {
        "Buruk": fuzz.trimf(u_s, [1, 1, 3]),
        "Cukup": fuzz.trimf(u_s, [2, 3, 4]),
        "Bagus": fuzz.trimf(u_s, [3, 5, 5]),
    }

    rows = []
    for _, row in df_input.iterrows():
        c1 = int(np.clip(row["C1_Waktu"],     0, 150))
        c2 = int(np.clip(row["C2_Biaya"],     0, 2000))
        c3 = int(np.clip(row["C3_Cuaca"],     1, 5))
        c4 = int(np.clip(row["C4_Kapasitas"], 1, 5))
        c5 = int(np.clip(row["C5_Medan"],     1, 5))
        rows.append({
            "ID":          row["ID_Alternatif"],
            "Moda":        row["Moda_Transportasi"],
            "C1 (mnt)":    c1,
            "µ Cepat":     round(fuzz.interp_membership(u_w, mf_w["Cepat"],  c1), 3),
            "µ Sedang":    round(fuzz.interp_membership(u_w, mf_w["Sedang"], c1), 3),
            "µ Lambat":    round(fuzz.interp_membership(u_w, mf_w["Lambat"], c1), 3),
            "C2 (rb Rp)":  c2,
            "µ Murah":     round(fuzz.interp_membership(u_b, mf_b["Murah"],   c2), 3),
            "µ Standar":   round(fuzz.interp_membership(u_b, mf_b["Standar"], c2), 3),
            "µ Mahal":     round(fuzz.interp_membership(u_b, mf_b["Mahal"],   c2), 3),
            "C3 (Cuaca)":  c3,
            "µ Cu-Buruk":  round(fuzz.interp_membership(u_s, mf_s["Buruk"], c3), 3),
            "µ Cu-Cukup":  round(fuzz.interp_membership(u_s, mf_s["Cukup"], c3), 3),
            "µ Cu-Bagus":  round(fuzz.interp_membership(u_s, mf_s["Bagus"], c3), 3),
            "C4 (Kap.)":   c4,
            "µ Ka-Buruk":  round(fuzz.interp_membership(u_s, mf_s["Buruk"], c4), 3),
            "µ Ka-Cukup":  round(fuzz.interp_membership(u_s, mf_s["Cukup"], c4), 3),
            "µ Ka-Bagus":  round(fuzz.interp_membership(u_s, mf_s["Bagus"], c4), 3),
            "C5 (Medan)":  c5,
            "µ Me-Buruk":  round(fuzz.interp_membership(u_s, mf_s["Buruk"], c5), 3),
            "µ Me-Cukup":  round(fuzz.interp_membership(u_s, mf_s["Cukup"], c5), 3),
            "µ Me-Bagus":  round(fuzz.interp_membership(u_s, mf_s["Bagus"], c5), 3),
        })
    return pd.DataFrame(rows)

# ============================================================
# 5. HEADER UTAMA
# ============================================================
st.title("🚑 Sistem Pendukung Keputusan")
st.subheader("Pemilihan Moda Transportasi Darurat Distribusi Obat ke Daerah Terpencil Saat Bencana")
st.caption("Metode: Fuzzy Mamdani  |  Dataset: 250 Alternatif  |  Kriteria: 5")
st.divider()

# ============================================================
# 6. TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset",
    "📉 Kurva Keanggotaan",
    "🔬 Proses Fuzzifikasi",
    "🏆 Hasil Ranking",
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — DATASET
# ─────────────────────────────────────────────────────────────
with tab1:
    st.write("### Data Alternatif Transportasi")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Data Asli",  len(df_raw))
    col_b.metric("Data Terfilter",   len(df_filtered))
    col_c.metric("Jenis Moda",       df_raw["Moda_Transportasi"].nunique())
    st.dataframe(df_filtered, width='stretch')

# ─────────────────────────────────────────────────────────────
# TAB 2 — KURVA KEANGGOTAAN
# ─────────────────────────────────────────────────────────────
with tab2:
    st.write("### Visualisasi Kurva Keanggotaan (Membership Function)")
    st.info("Setiap kurva menggambarkan bagaimana nilai input dikonversi menjadi derajat keanggotaan fuzzy (skala 0–1).")
    try:
        v_w, v_b, v_cu, v_ka, v_me, v_sk = buat_fuzzy_vars()
        COLORS = ["#e74c3c", "#f39c12", "#27ae60"]
        var_info = [
            (v_w,  "C1: Waktu (Menit)"),
            (v_b,  "C2: Biaya (Ribu Rp)"),
            (v_cu, "C3: Cuaca (Skala 1–5)"),
            (v_ka, "C4: Kapasitas (Skala 1–5)"),
            (v_me, "C5: Medan (Skala 1–5)"),
            (v_sk, "Output: Skor (0–100)"),
        ]
        cols = st.columns(3)
        for i, (var, title) in enumerate(var_info):
            with cols[i % 3]:
                fig, ax = plt.subplots(figsize=(5, 3))
                for ci, lbl in enumerate(var.terms):
                    ax.plot(var.universe, var[lbl].mf,
                            label=lbl, linewidth=2.2, color=COLORS[ci % 3])
                ax.set_title(title, fontsize=9, fontweight="bold")
                ax.legend(fontsize=8)
                ax.set_ylim(-0.05, 1.15)
                ax.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig)
                plt.close(fig)
    except Exception as e:
        st.error(f"Gagal menampilkan kurva: {e}")

# ─────────────────────────────────────────────────────────────
# TAB 3 — PROSES FUZZIFIKASI
# ─────────────────────────────────────────────────────────────
with tab3:
    st.write("### Tabel Proses Fuzzifikasi")
    st.info(
        "Tabel di bawah menampilkan **derajat keanggotaan (µ)** setiap nilai input "
        "terhadap himpunan fuzzy masing-masing variabel kriteria. "
        "Nilai µ berkisar antara 0 (tidak anggota) hingga 1 (anggota penuh)."
    )
    try:
        df_fuzz = hitung_fuzzifikasi(df_filtered)
        sections = [
            ("C1 — Waktu",     ["ID", "Moda", "C1 (mnt)",   "µ Cepat",    "µ Sedang",   "µ Lambat"]),
            ("C2 — Biaya",     ["ID", "Moda", "C2 (rb Rp)", "µ Murah",    "µ Standar",  "µ Mahal"]),
            ("C3 — Cuaca",     ["ID", "Moda", "C3 (Cuaca)", "µ Cu-Buruk", "µ Cu-Cukup", "µ Cu-Bagus"]),
            ("C4 — Kapasitas", ["ID", "Moda", "C4 (Kap.)",  "µ Ka-Buruk", "µ Ka-Cukup", "µ Ka-Bagus"]),
            ("C5 — Medan",     ["ID", "Moda", "C5 (Medan)", "µ Me-Buruk", "µ Me-Cukup", "µ Me-Bagus"]),
        ]
        for idx, (label, cols) in enumerate(sections):
            with st.expander(f"📋 {label}", expanded=(idx == 0)):
                st.dataframe(df_fuzz[cols], width='stretch')
    except Exception as e:
        st.error(f"Tabel fuzzifikasi gagal: {e}")

# ─────────────────────────────────────────────────────────────
# TAB 4 — HASIL RANKING
# ─────────────────────────────────────────────────────────────
with tab4:
    st.write("### Hasil Perangkingan SPK Fuzzy Mamdani")
    st.write(
        f"**Konfigurasi saat ini:** Threshold ≥ {threshold:.1f}  |  "
        f"Filter = {filter_moda}  |  Top = {top_n}"
    )

    if st.button("🚀 Jalankan Perhitungan SPK", type="primary"):
        with st.spinner("⏳ Menghitung skor Fuzzy Mamdani untuk semua alternatif..."):
            try:
                skor_list = hitung_skor_fuzzy(df_filtered)
                df_hasil = df_filtered.copy()
                df_hasil["Skor_Mamdani"] = skor_list

                df_sorted = (
                    df_hasil[df_hasil["Skor_Mamdani"] >= threshold]
                    .sort_values("Skor_Mamdani", ascending=False)
                    .reset_index(drop=True)
                )
                df_sorted.index += 1
                df_sorted.index.name = "Peringkat"

                if df_sorted.empty:
                    st.warning(
                        f"⚠️ Tidak ada alternatif dengan skor ≥ {threshold:.1f}. "
                        "Coba turunkan nilai threshold di sidebar."
                    )
                else:
                    st.success(f"✅ Ditemukan **{len(df_sorted)}** alternatif yang lolos threshold.")

                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🥇 Skor Tertinggi",        f"{df_sorted['Skor_Mamdani'].max():.2f}")
                    m2.metric("📊 Rata-rata Skor",         f"{df_sorted['Skor_Mamdani'].mean():.2f}")
                    m3.metric("🥉 Skor Terendah (lolos)",  f"{df_sorted['Skor_Mamdani'].min():.2f}")

                    # Tabel Perangkingan
                    st.write("#### Tabel Hasil Perangkingan")
                    cols_show = [
                        "ID_Alternatif", "Moda_Transportasi",
                        "C1_Waktu", "C2_Biaya", "C3_Cuaca",
                        "C4_Kapasitas", "C5_Medan", "Skor_Mamdani"
                    ]
                    st.dataframe(df_sorted[cols_show], width='stretch')

                    # Bar chart Top N
                    st.write(f"#### 📊 Grafik Top {top_n} Alternatif Terbaik")
                    top_data = df_sorted.head(top_n)
                    labels = (
                        top_data["ID_Alternatif"] + " — " + top_data["Moda_Transportasi"]
                    ).tolist()

                    fig_bar, ax_bar = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
                    bar_colors = plt.cm.RdYlGn(np.linspace(0.35, 0.85, len(top_data)))
                    bars = ax_bar.barh(labels, top_data["Skor_Mamdani"].values, color=bar_colors)
                    for bar, val in zip(bars, top_data["Skor_Mamdani"].values):
                        ax_bar.text(
                            bar.get_width() + 0.3,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.2f}", va="center", fontsize=8
                        )
                    ax_bar.set_xlabel("Skor Mamdani", fontsize=11)
                    ax_bar.set_title(
                        f"Top {top_n} Alternatif Terbaik (Fuzzy Mamdani)",
                        fontsize=12, fontweight="bold"
                    )
                    ax_bar.invert_yaxis()
                    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)

                    # Pie chart distribusi moda
                    st.write("#### 🥧 Distribusi Moda Transportasi (Lolos Threshold)")
                    moda_count = df_sorted["Moda_Transportasi"].value_counts()
                    fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
                    ax_pie.pie(
                        moda_count.values,
                        labels=moda_count.index,
                        autopct="%1.1f%%",
                        startangle=90,
                        colors=plt.cm.Set3(np.linspace(0, 1, len(moda_count)))
                    )
                    ax_pie.set_title(
                        "Distribusi Moda Transportasi Terpilih",
                        fontsize=11, fontweight="bold"
                    )
                    st.pyplot(fig_pie)
                    plt.close(fig_pie)

            except Exception as e:
                st.error(f"Kesalahan saat perhitungan: {e}")
    else:
        st.info("👆 Klik tombol di atas untuk memulai proses inferensi Fuzzy Mamdani.")