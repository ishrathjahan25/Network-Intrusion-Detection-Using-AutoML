import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Network Intrusion Detection - AutoML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        border: 1px solid #e94560;
    }
    .main-header h1 { color: #e94560; font-size: 2rem; margin: 0; }
    .main-header p  { color: #a0aec0; margin: 0.5rem 0 0; }
    .metric-card {
        background: #1a1a2e; border: 1px solid #2d3748;
        border-radius: 10px; padding: 1.2rem; text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #e94560; }
    .metric-card .label { font-size: 0.85rem; color: #a0aec0; margin-top: 0.3rem; }
    .attack-badge {
        background: #e94560; color: white; padding: 0.6rem 1.5rem;
        border-radius: 50px; font-size: 1.4rem; font-weight: 700;
        display: inline-block; margin: 0.5rem 0;
    }
    .benign-badge {
        background: #38a169; color: white; padding: 0.6rem 1.5rem;
        border-radius: 50px; font-size: 1.4rem; font-weight: 700;
        display: inline-block; margin: 0.5rem 0;
    }
    .section-header {
        color: #e94560; font-size: 1.2rem; font-weight: 600;
        border-bottom: 2px solid #e94560; padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.95rem; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Network Intrusion Detection Using AutoML</h1>
    <p>BSc (Hons) Cyber Security &nbsp;|&nbsp; De Montfort University Dubai &nbsp;|&nbsp;
       Ishrath Jahan Basheer - P2836139 &nbsp;|&nbsp; Supervisor: Dr. Sayed Ahmad</p>
</div>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = BASE_DIR  
csv_path = os.path.join(BASE, "data", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
df = pd.read_csv(csv_path)

# ── Load RF model (or retrain) ────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_rf_model():
    model_path = os.path.join(BASE, "rf_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    # retrain if not saved
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(
        os.path.join(BASE, "data",
                     "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
        low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df["Label"] = df["Label"].apply(lambda x: 0 if x.strip() == "BENIGN" else 1)
    X = df.drop("Label", axis=1)
    y = df["Label"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    return rf

# ── Tabs ──────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "📊 Model Performance",
    "🏆 AutoML Leaderboard",
    "🔍 SHAP Explainability",
    "🔮 Live Prediction"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Project Summary</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="value">286,096</div><div class="label">Total Records</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="value">78</div><div class="label">Features</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="value">14</div><div class="label">AutoML Models Trained</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="value">1.0000</div><div class="label">Best ROC-AUC</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-header">Dataset: CICIDS2017 — PortScan</div>', unsafe_allow_html=True)
        info = {
            "Property": ["Dataset", "File", "Raw Records", "Clean Records",
                          "Train Set (80%)", "Test Set (20%)", "Features",
                          "Attack Type", "Primary AutoML Tool"],
            "Value": ["CICIDS2017", "Friday-Afternoon-PortScan", "286,467",
                      "286,096", "228,876", "57,220", "78 numerical flow-level",
                      "PortScan", "H2O AutoML (GBM)"]
        }
        st.dataframe(pd.DataFrame(info), hide_index=True, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        labels = ["BENIGN\n(127,292)", "PortScan\n(158,804)"]
        sizes  = [127292, 158804]
        colors = ["#38a169", "#e94560"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"color": "white", "fontsize": 11})
        for at in autotexts:
            at.set_color("white"); at.set_fontsize(12)
        ax.set_title("Label Distribution", color="white", fontsize=13, pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-header">Research Questions</div>', unsafe_allow_html=True)
    rq_data = {
        "RQ": ["RQ1", "RQ2", "RQ3"],
        "Question": [
            "Does AutoML perform better than Logistic Regression and Random Forest?",
            "What is the trade-off between performance and computational cost?",
            "Can SHAP explainability identify plausible network features?"
        ],
        "Finding": [
            " H2O AutoML matched RF with only 5 total errors vs 50 for LR",
            " AutoML explored 14 models automatically at cost of 665s vs 17s for RF",
            " PSH Flag Count, packet size features align with PortScan behaviour"
        ]
    }
    st.dataframe(pd.DataFrame(rq_data), hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Model Comparison Summary</div>', unsafe_allow_html=True)

    perf_data = {
        "Model": ["Logistic Regression", "Random Forest", "H2O AutoML (GBM_1)"],
        "Precision": [0.9991, 0.9999, 0.9999],
        "Recall":    [0.9991, 0.9999, 0.9999],
        "F1-Score":  [0.9991, 0.9999, 0.9999],
        "ROC-AUC":   [0.9999, 1.0000, 1.0000],
        "Train Time (s)": [2.5, 17.3, 665.4],
        "Total Errors": [50, 5, 5]
    }
    df_perf = pd.DataFrame(perf_data)
    st.dataframe(df_perf.style.highlight_max(
        subset=["Precision","Recall","F1-Score","ROC-AUC"], color="#1a4731"
    ).highlight_min(
        subset=["Total Errors"], color="#1a4731"
    ), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)

    cm_files = {
        "Logistic Regression": "confusion_matrix_lr.png",
        "Random Forest":       "confusion_matrix_rf.png",
        "H2O AutoML (GBM)":   "confusion_matrix_h2o.png"
    }
    cols = st.columns(3)
    for i, (name, fname) in enumerate(cm_files.items()):
        path = os.path.join(BASE, fname)
        with cols[i]:
            st.markdown(f"**{name}**")
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning(f"File not found: {fname}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Error Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        models = ["Logistic\nRegression", "Random\nForest", "H2O AutoML\n(GBM)"]
        errors = [50, 5, 5]
        bars = ax.bar(models, errors, color=["#e94560", "#38a169", "#4299e1"],
                      edgecolor="none", width=0.5)
        ax.set_ylabel("Total Errors", color="white")
        ax.set_title("Total Prediction Errors on Test Set", color="white", fontsize=12)
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_facecolor("#1a1a2e")
        for bar, val in zip(bars, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha="center", color="white", fontsize=12, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        fn = [28, 4, 4]
        fp = [22, 1, 1]
        x = np.arange(3)
        w = 0.35
        ax.bar(x - w/2, fn, w, label="False Negatives (Missed Attacks)",
               color="#e94560", edgecolor="none")
        ax.bar(x + w/2, fp, w, label="False Positives (False Alarms)",
               color="#ed8936", edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(["LR", "RF", "H2O AutoML"], color="white")
        ax.set_ylabel("Count", color="white")
        ax.set_title("False Negatives vs False Positives", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ════════════════════════════════════════════════════════════
# TAB 3 — AUTOML LEADERBOARD
# ════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">H2O AutoML Leaderboard — All 14 Models</div>', unsafe_allow_html=True)

    lb_data = {
        "Rank": list(range(1, 15)),
        "Model": [
            "GBM_grid_1_model_3 ⭐", "GBM_3", "GBM_5", "GBM_2", "GBM_4",
            "GBM_1 (Leader)", "GBM_grid_1_model_5", "GBM_grid_1_model_1",
            "GBM_grid_1_model_4", "DRF_1", "XRT_1", "GBM_grid_1_model_2",
            "DeepLearning_1", "GLM_1"
        ],
        "AUC":      [1.0]*12 + [0.999953, 0.999926],
        "LogLoss":  [0.000442, 0.000594, 0.000511, 0.000503, 0.000600,
                     0.000499, 0.000802, 0.000389, 0.000577, 0.002929,
                     0.115644, 0.000797, 0.006048, 0.006299],
        "AUCPR":    [1.0]*12 + [0.999974, 0.999960],
        "RMSE":     [0.01113, 0.01327, 0.01200, 0.01242, 0.01248,
                     0.01222, 0.01320, 0.00985, 0.01220, 0.01239,
                     0.13950, 0.01227, 0.02575, 0.03068]
    }
    df_lb = pd.DataFrame(lb_data)
    st.dataframe(df_lb.style.highlight_min(subset=["LogLoss", "RMSE"], color="#1a4731"),
                 hide_index=True, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">LogLoss by Model (Lower = Better)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        short_names = [m.replace("_AutoML_1_20260530_211236","").replace("GBM_grid_1_","Grid_")
                       for m in lb_data["Model"]]
        colors_lb = ["#e94560" if "⭐" in n or "Leader" in n else "#4299e1"
                     for n in lb_data["Model"]]
        bars = ax.barh(short_names, lb_data["LogLoss"], color=colors_lb, edgecolor="none")
        ax.set_xlabel("LogLoss", color="white")
        ax.tick_params(colors="white", labelsize=8)
        ax.set_title("AutoML Leaderboard — LogLoss", color="white", fontsize=11)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Algorithm Family Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        families = {"GBM": 9, "DRF": 1, "XRT": 1, "Deep Learning": 1, "GLM": 1}
        colors_f = ["#e94560", "#38a169", "#4299e1", "#ed8936", "#9f7aea"]
        wedges, texts, autotexts = ax.pie(
            families.values(), labels=families.keys(),
            colors=colors_f, autopct="%1.0f%%",
            textprops={"color": "white", "fontsize": 10})
        for at in autotexts:
            at.set_color("white")
        ax.set_title("Models by Algorithm Family", color="white", fontsize=11)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.info("⭐ **Best model by validation LogLoss:** GBM_grid_1_model_3 (LogLoss = 0.000442, RMSE = 0.01113). H2O selected GBM_1 as leader based on cross-validation AUC.")

# ════════════════════════════════════════════════════════════
# TAB 4 — SHAP
# ════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">SHAP Feature Importance Analysis (Random Forest)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        shap_path = os.path.join(BASE, "shap_summary.png")
        if os.path.exists(shap_path):
            st.image(shap_path, caption="Figure 13: SHAP Beeswarm Summary Plot — Top 15 Features",
                     use_container_width=True)
        else:
            st.warning("shap_summary.png not found in project folder.")

    with col2:
        st.markdown('<div class="section-header">Top 10 Features by Mean |SHAP|</div>', unsafe_allow_html=True)
        shap_data = {
            "Rank": list(range(1, 11)),
            "Feature": [
                "PSH Flag Count", "Subflow Fwd Bytes",
                "Fwd Packet Length Mean", "Average Packet Size",
                "Total Length of Fwd Packets", "Fwd Packet Length Max",
                "Bwd Packets/s", "Packet Length Mean",
                "Avg Fwd Segment Size", "Init_Win_bytes_forward"
            ],
            "Mean |SHAP|": [0.0431, 0.0311, 0.0310, 0.0302, 0.0289,
                             0.0256, 0.0249, 0.0245, 0.0237, 0.0220]
        }
        df_shap = pd.DataFrame(shap_data)
        st.dataframe(df_shap, hide_index=True, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5, 5), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        y_pos = range(len(df_shap))
        ax.barh(list(y_pos), df_shap["Mean |SHAP|"], color="#e94560", edgecolor="none")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(df_shap["Feature"], color="white", fontsize=8)
        ax.set_xlabel("Mean |SHAP| Value", color="white")
        ax.tick_params(colors="white")
        ax.set_title("Feature Importance (SHAP)", color="white", fontsize=10)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">Feature Interpretation</div>', unsafe_allow_html=True)
    interp = {
        "Feature": ["PSH Flag Count", "Subflow Fwd Bytes / Fwd Packet Length",
                    "Average Packet Size", "Bwd Packets/s", "Init_Win_bytes_forward"],
        "Why It Matters for PortScan Detection": [
            "PortScan probes use unusual TCP flag combinations — high PSH count is a strong attack signal",
            "Port scans send minimal-payload packets → very low forward byte counts → strong ATTACK signal",
            "Scan traffic has tiny packets (single SYN) → abnormally low average packet size",
            "Scanners rarely get responses → backward traffic rate near zero for attack flows",
            "Initial TCP window size differs between legitimate connections and SYN scan probes"
        ]
    }
    st.dataframe(pd.DataFrame(interp), hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 5 — LIVE PREDICTION
# ════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">🔮 Live Network Flow Prediction</div>', unsafe_allow_html=True)
    st.markdown("Adjust the network flow features below and click **Predict** to classify a flow as BENIGN or ATTACK.")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("**📦 Packet Size Features**")
        fwd_pkt_len_mean   = st.slider("Fwd Packet Length Mean",   0.0, 1500.0, 64.97)
        avg_pkt_size       = st.slider("Average Packet Size",       0.0, 1500.0, 64.97)
        pkt_len_mean       = st.slider("Packet Length Mean",        0.0, 1500.0, 50.0)
        fwd_pkt_len_max    = st.slider("Fwd Packet Length Max",     0.0, 1500.0, 456.0)
        total_fwd_bytes    = st.slider("Total Length of Fwd Packets", 0.0, 100000.0, 2664.0)
        subflow_fwd_bytes  = st.slider("Subflow Fwd Bytes",         0.0, 100000.0, 2664.0)
        avg_fwd_seg        = st.slider("Avg Fwd Segment Size",      0.0, 1500.0, 64.97)

        st.markdown("**⏱️ Timing Features**")
        flow_duration  = st.slider("Flow Duration (μs)",     0, 5000000, 1266342)
        flow_iat_max   = st.slider("Flow IAT Max (μs)",      0, 5000000, 500000)
        bwd_pkts_s     = st.slider("Bwd Packets/s",          0.0, 10000.0, 31.5)

    with col_r:
        st.markdown("**🚩 TCP Flag Features**")
        psh_flag_count = st.slider("PSH Flag Count",   0, 100, 0)
        ack_flag_count = st.slider("ACK Flag Count",   0, 100, 41)
        rst_flag_count = st.slider("RST Flag Count",   0, 100, 0)
        syn_flag_count = st.slider("SYN Flag Count",   0, 100, 0)
        fin_flag_count = st.slider("FIN Flag Count",   0, 100, 0)

        st.markdown("**📊 Flow Packet Counts**")
        total_fwd_pkts    = st.slider("Total Fwd Packets",      0, 200, 41)
        total_bwd_pkts    = st.slider("Total Backward Packets", 0, 200, 44)
        subflow_fwd_pkts  = st.slider("Subflow Fwd Packets",    0, 200, 41)
        init_win_forward  = st.slider("Init_Win_bytes_forward", 0, 65535, 29200)

    st.markdown("<br>", unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        predict_btn = st.button("🔍 Predict This Flow", type="primary", use_container_width=True)
    with btn_col2:
        attack_preset_btn = st.button("🚨 Predict Real Attack Sample", type="secondary", use_container_width=True)

    if predict_btn or attack_preset_btn:
        with st.spinner("Loading model and predicting..."):
            try:
                rf_model = load_rf_model()
                feature_names = rf_model.feature_names_in_ if hasattr(rf_model, "feature_names_in_") else None

                if attack_preset_btn:
                    # Load a real attack row directly from test set
                    import pandas as pd
                    df_raw = pd.read_csv(
                        os.path.join(BASE, "data",
                                     "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
                        low_memory=False)
                    df_raw.columns = df_raw.columns.str.strip()
                    df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df_raw.dropna(inplace=True)
                    df_raw["Label"] = df_raw["Label"].apply(lambda x: 0 if x.strip() == "BENIGN" else 1)
                    from sklearn.model_selection import train_test_split
                    X_all = df_raw.drop("Label", axis=1)
                    y_all = df_raw["Label"]
                    _, X_t, _, y_t = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
                    attack_sample = X_t[y_t == 1].iloc[[0]]
                    sample = attack_sample
                    st.info("Using a real PortScan record from the test set (all 78 features set correctly).")

                else:
                    if feature_names is not None:
                        sample = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
                        feature_map = {
                            "Fwd Packet Length Mean":       fwd_pkt_len_mean,
                            "Average Packet Size":          avg_pkt_size,
                            "Packet Length Mean":           pkt_len_mean,
                            "Fwd Packet Length Max":        fwd_pkt_len_max,
                            "Total Length of Fwd Packets":  total_fwd_bytes,
                            "Subflow Fwd Bytes":            subflow_fwd_bytes,
                            "Avg Fwd Segment Size":         avg_fwd_seg,
                            "Flow Duration":                flow_duration,
                            "Flow IAT Max":                 flow_iat_max,
                            "Bwd Packets/s":                bwd_pkts_s,
                            "PSH Flag Count":               psh_flag_count,
                            "ACK Flag Count":               ack_flag_count,
                            "RST Flag Count":               rst_flag_count,
                            "SYN Flag Count":               syn_flag_count,
                            "FIN Flag Count":               fin_flag_count,
                            "Total Fwd Packets":            total_fwd_pkts,
                            "Total Backward Packets":       total_bwd_pkts,
                            "Subflow Fwd Packets":          subflow_fwd_pkts,
                            "Init_Win_bytes_forward":       init_win_forward,
                        }
                        for feat, val in feature_map.items():
                            if feat in sample.columns:
                                sample[feat] = val

                pred  = rf_model.predict(sample)[0]
                proba = rf_model.predict_proba(sample)[0]

                res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
                with res_col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if pred == 1:
                        st.markdown('<div style="text-align:center"><div class="attack-badge">🚨 ATTACK DETECTED</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="text-align:center; color:#a0aec0; margin-top:0.5rem">Confidence: <b style="color:#e94560">{proba[1]*100:.1f}%</b> Attack &nbsp;|&nbsp; <b>{proba[0]*100:.1f}%</b> Benign</div>', unsafe_allow_html=True)
                        st.error("⚠️ This network flow exhibits characteristics consistent with a PortScan attack.")
                    else:
                        st.markdown('<div style="text-align:center"><div class="benign-badge"> BENIGN TRAFFIC</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="text-align:center; color:#a0aec0; margin-top:0.5rem">Confidence: <b style="color:#38a169">{proba[0]*100:.1f}%</b> Benign &nbsp;|&nbsp; <b>{proba[1]*100:.1f}%</b> Attack</div>', unsafe_allow_html=True)
                        st.success("This network flow appears to be normal legitimate traffic.")

                    st.markdown("<br>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(5, 1.5), facecolor="#1a1a2e")
                    ax.set_facecolor("#1a1a2e")
                    ax.barh(["Attack", "Benign"], [proba[1], proba[0]],
                            color=["#e94560", "#38a169"], edgecolor="none", height=0.5)
                    ax.set_xlim(0, 1)
                    ax.tick_params(colors="white")
                    ax.set_xlabel("Probability", color="white")
                    for spine in ax.spines.values(): spine.set_visible(False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Make sure rf_model.pkl and the dataset CSV are in the same folder as app.py.")
# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#718096; font-size:0.8rem">'
    'Network Intrusion Detection Using AutoML &nbsp;|&nbsp; '
    'Ishrath Jahan Basheer P2836139 &nbsp;|&nbsp; '
    'BSc (Hons) Cyber Security &nbsp;|&nbsp; De Montfort University Dubai 2023-26'
    '</p>',
    unsafe_allow_html=True
)
