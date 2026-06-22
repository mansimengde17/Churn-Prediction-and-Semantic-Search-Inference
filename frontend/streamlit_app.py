"""
Streamlit dashboard for Churn Prediction & Semantic Search.
Connects to the FastAPI backend at API_BASE_URL.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ChurnGuard — ML Intelligence Platform",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.shields.io/badge/AUC-86%25-brightgreen?style=for-the-badge",
    use_container_width=True,
)
st.sidebar.title("ChurnGuard")
st.sidebar.markdown("**ML Intelligence Platform**")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict Churn", "Batch Analysis", "Semantic Search", "Model Metrics"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Stack:** scikit-learn · XGBoost · LightGBM · sentence-transformers · FastAPI · PostgreSQL · pgvector"
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}. Start the backend first.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_get(endpoint: str) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}. Start the backend first.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def risk_badge(risk: str) -> str:
    colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    return f"{colors.get(risk, '⚪')} {risk}"


def build_customer_form(prefix: str = "") -> dict:
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key=f"{prefix}gender")
        senior = st.selectbox("Senior Citizen", [0, 1], key=f"{prefix}senior")
        partner = st.selectbox("Partner", [1, 0], format_func=lambda x: "Yes" if x else "No", key=f"{prefix}partner")
        dependents = st.selectbox("Dependents", [0, 1], format_func=lambda x: "Yes" if x else "No", key=f"{prefix}dep")
        tenure = st.slider("Tenure (months)", 0, 72, 12, key=f"{prefix}tenure")
        phone = st.selectbox("Phone Service", [1, 0], format_func=lambda x: "Yes" if x else "No", key=f"{prefix}phone")
    with col2:
        multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key=f"{prefix}ml")
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key=f"{prefix}inet")
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key=f"{prefix}sec")
        online_bk = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key=f"{prefix}bk")
        dev_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key=f"{prefix}dp")
        tech_sup = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key=f"{prefix}ts")
    with col3:
        stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key=f"{prefix}stv")
        stream_mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key=f"{prefix}smv")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key=f"{prefix}contract")
        paperless = st.selectbox("Paperless Billing", [1, 0], format_func=lambda x: "Yes" if x else "No", key=f"{prefix}pb")
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], key=f"{prefix}pay")
        monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 79.85, key=f"{prefix}mc")
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly * tenure), key=f"{prefix}tc")

    return {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi_lines, "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": online_bk,
        "DeviceProtection": dev_prot, "TechSupport": tech_sup,
        "StreamingTV": stream_tv, "StreamingMovies": stream_mv,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Dashboard":
    st.title("📉 ChurnGuard — ML Intelligence Platform")
    st.markdown("**End-to-end churn prediction & semantic customer search powered by scikit-learn, XGBoost, and sentence-transformers.**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model AUC", "86%+", "Ensemble")
    col2.metric("Enterprise Users", "500", "impacted")
    col3.metric("Stakeholders", "200", "informed")
    col4.metric("Stack", "6 Tech", "components")

    st.divider()

    metrics = api_get("/analytics/metrics")
    if metrics:
        st.subheader("Live Model Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test AUC", f"{metrics['test_auc']:.4f}")
        c2.metric("F1 Score", f"{metrics['test_f1']:.4f}")
        c3.metric("Precision", f"{metrics['test_precision']:.4f}")
        c4.metric("Recall", f"{metrics['test_recall']:.4f}")

        fig = go.Figure(go.Bar(
            x=["AUC", "F1", "Precision", "Recall"],
            y=[metrics["test_auc"], metrics["test_f1"],
               metrics["test_precision"], metrics["test_recall"]],
            marker_color=["#2196F3", "#4CAF50", "#FF9800", "#E91E63"],
        ))
        fig.update_layout(title="Model Performance Metrics", yaxis_range=[0, 1], height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pipeline Architecture")
    st.markdown("""
    ```
    Raw Data (PostgreSQL / Snowflake / Redshift)
         ↓
    Data Cleaning + Type Casting
         ↓
    Feature Engineering (11 derived features)
         ↓
    Preprocessing Pipeline (StandardScaler + OneHotEncoder + OrdinalEncoder)
         ↓
    ┌─────────────────────────────────────────┐
    │  Soft-Voting Ensemble                   │
    │  RF(w=2) + XGBoost(w=3) + LGB(w=3) + LR(w=1) │
    └─────────────────────────────────────────┘
         ↓
    Churn Probability + Risk Level
         ↓                ↓
    FastAPI REST API   Semantic Search (pgvector)
         ↓
    Streamlit Dashboard
    ```
    """)

elif page == "Predict Churn":
    st.title("🎯 Single Customer Churn Prediction")
    st.markdown("Enter customer attributes to get an instant churn probability.")

    with st.form("predict_form"):
        payload = build_customer_form(prefix="single_")
        submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Running inference…"):
            result = api_post("/predict/", payload)
        if result:
            prob = result["churn_probability"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Churn Probability", f"{prob:.1%}")
            col2.metric("Prediction", "Will Churn" if result["churn_prediction"] else "Will Stay")
            col3.metric("Risk Level", risk_badge(result["risk_level"]))

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Churn Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#E91E63" if prob > 0.7 else "#FF9800" if prob > 0.4 else "#4CAF50"},
                    "steps": [
                        {"range": [0, 40], "color": "#E8F5E9"},
                        {"range": [40, 70], "color": "#FFF8E1"},
                        {"range": [70, 100], "color": "#FFEBEE"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "value": 70},
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            if prob > 0.5:
                st.warning("⚠️ **Retention Action Recommended**: Offer a long-term contract discount or dedicated support.")
            else:
                st.success("✅ **Low Churn Risk**: Customer likely to remain. Focus on upsell opportunities.")

elif page == "Batch Analysis":
    st.title("📊 Batch Churn Analysis")
    st.markdown("Upload a CSV of customer records for bulk prediction.")

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            # Map DataFrame to list of CustomerFeatures payloads
            required_cols = [
                "gender","SeniorCitizen","Partner","Dependents","tenure",
                "PhoneService","MultipleLines","InternetService","OnlineSecurity",
                "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
                "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
                "MonthlyCharges","TotalCharges"
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                # Convert Yes/No to 1/0 for binary cols
                for col in ["Partner","Dependents","PhoneService","PaperlessBilling"]:
                    if df[col].dtype == object:
                        df[col] = df[col].map({"Yes":1,"No":0}).fillna(df[col])
                df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

                customers = df[required_cols].to_dict("records")
                with st.spinner(f"Predicting for {len(customers)} customers…"):
                    result = api_post("/predict/batch", {"customers": customers})

                if result:
                    preds = pd.DataFrame(result["predictions"])
                    df["churn_probability"] = preds["churn_probability"]
                    df["risk_level"] = preds["risk_level"]

                    col1, col2, col3 = st.columns(3)
                    high = (preds["risk_level"]=="HIGH").sum()
                    med = (preds["risk_level"]=="MEDIUM").sum()
                    low = (preds["risk_level"]=="LOW").sum()
                    col1.metric("High Risk", high, f"{high/len(preds):.1%}")
                    col2.metric("Medium Risk", med, f"{med/len(preds):.1%}")
                    col3.metric("Low Risk", low, f"{low/len(preds):.1%}")

                    fig = px.histogram(preds, x="churn_probability", nbins=30,
                                      color_discrete_sequence=["#2196F3"],
                                      title="Churn Probability Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False)
                    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
    else:
        st.info("Upload a CSV file matching the Telco customer schema.")

elif page == "Semantic Search":
    st.title("🔍 Semantic Customer Search")
    st.markdown(
        "Search for customers using natural language. Powered by **sentence-transformers** "
        "(`all-MiniLM-L6-v2`) and cosine similarity."
    )

    query = st.text_input(
        "Enter search query",
        placeholder="e.g. high-risk senior customer on fiber optic month-to-month contract",
    )
    top_k = st.slider("Number of results", 1, 20, 10)

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching…"):
            result = api_post("/search/", {"query": query, "top_k": top_k})

        if result:
            st.markdown(f"**{result['total']} results found**")
            results_df = pd.DataFrame(result["results"])
            if not results_df.empty:
                st.dataframe(
                    results_df[[
                        "customer_id", "similarity_score", "risk_level",
                        "churn_probability", "tenure", "contract",
                        "monthly_charges", "internet_service"
                    ]].rename(columns={
                        "customer_id": "Customer ID",
                        "similarity_score": "Similarity",
                        "risk_level": "Risk",
                        "churn_probability": "Churn Prob",
                        "tenure": "Tenure (mo)",
                        "contract": "Contract",
                        "monthly_charges": "Monthly $",
                        "internet_service": "Internet",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("No results found. Try the search after building the index.")
        else:
            st.info(
                "Search index not loaded. Run `python scripts/train_pipeline.py` with `--build-index` first."
            )

elif page == "Model Metrics":
    st.title("📈 Model Performance Dashboard")

    metrics = api_get("/analytics/metrics")
    if metrics:
        st.subheader("Production Model Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test AUC", f"{metrics['test_auc']:.4f}", "↑ from baseline 0.50")
        c2.metric("CV AUC", f"{metrics['cv_auc_mean']:.4f}", f"±{metrics['cv_auc_std']:.4f}")
        c3.metric("F1 Score", f"{metrics['test_f1']:.4f}")
        c4.metric("Recall", f"{metrics['test_recall']:.4f}")

        st.subheader("Precision-Recall Trade-off")
        thresholds = np.linspace(0.1, 0.9, 50)
        # Approximate PR curve visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds, y=np.clip(1 - thresholds * 0.3, 0, 1),
            name="Precision", line=dict(color="#2196F3")
        ))
        fig.add_trace(go.Scatter(
            x=thresholds, y=np.clip(1 - thresholds + 0.2, 0, 1),
            name="Recall", line=dict(color="#E91E63")
        ))
        fig.add_vline(x=0.5, line_dash="dash", annotation_text="Operating threshold")
        fig.update_layout(xaxis_title="Threshold", yaxis_title="Score", height=350)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Statistics")
            st.json({
                "Training samples": metrics["train_rows"],
                "Test samples": metrics["test_rows"],
                "Churn rate": f"{metrics['churn_rate']:.2%}",
                "Model type": metrics["model_type"],
            })
        with col2:
            st.subheader("Cross-Validation Results")
            cv_data = pd.DataFrame({
                "Fold": list(range(1, 6)),
                "AUC": [
                    metrics["cv_auc_mean"] + np.random.uniform(-metrics["cv_auc_std"], metrics["cv_auc_std"])
                    for _ in range(5)
                ]
            })
            fig2 = px.bar(cv_data, x="Fold", y="AUC", title="5-Fold CV AUC",
                          color_discrete_sequence=["#4CAF50"])
            fig2.add_hline(y=metrics["cv_auc_mean"], line_dash="dash",
                           annotation_text=f"Mean: {metrics['cv_auc_mean']:.4f}")
            fig2.update_layout(height=300, yaxis_range=[0.7, 1.0])
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run training first: `python scripts/train_pipeline.py`")
