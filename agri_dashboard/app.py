import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Agri Renewable Dashboard", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("agri_energy.csv")
    df["year_month"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str))
    df["total_renewable_mw"] = df["solar_mw_district"] + df["wind_mw"] + df["bioenergy_mw"]
    df["co2_per_mw"] = df["estimated_co2_reduction_tonnes"] / (df["total_renewable_mw"] + 0.01)
    return df

df = load_data()
@st.cache_resource
def train_model(df, n_estimators: int, max_depth: int | None):

    features = [
        "state", "year", "month",
        "solar_mw_district", "wind_mw",
        "bioenergy_mw", "solar_pumps_installed"
    ]

    X = df[features]
    y = df["estimated_co2_reduction_tonnes"]

    cat_features = ["state"]
    num_features = [f for f in features if f != "state"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt"
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)

    return pipe, r2


# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filters")

states = ["All"] + sorted(df["state"].unique().tolist())
selected_state = st.sidebar.selectbox("Select State", states)

years = sorted(df["year"].unique().tolist())
selected_years = st.sidebar.multiselect("Select Years", years, default=years)

filtered = df[df["year"].isin(selected_years)]

if selected_state != "All":
    filtered = filtered[filtered["state"] == selected_state]

# ---------------- TITLE ----------------
st.title("Renewable Energy & Climate Mitigation in Indian Agriculture")

visual_tab, predict_tab = st.tabs(["Visuals", "Prediction"])

with visual_tab:
    # ---------------- KPIs ----------------
    c1, c2, c3 = st.columns(3)

    c1.metric("Total CO₂ Reduced (tonnes)", f"{filtered['estimated_co2_reduction_tonnes'].sum():,.0f}")
    c2.metric("Total Renewable Capacity (MW)", f"{filtered['total_renewable_mw'].sum():,.1f}")
    c3.metric("Solar Pumps Installed", f"{filtered['solar_pumps_installed'].sum():,.0f}")

    # ---------------- TIME SERIES ----------------
    st.subheader(" CO₂ Reduction Trend Over Time")

    trend = filtered.groupby("year_month")["estimated_co2_reduction_tonnes"].sum().reset_index()

    fig = px.line(trend, x="year_month", y="estimated_co2_reduction_tonnes",
                  labels={"year_month": "Time", "estimated_co2_reduction_tonnes": "CO₂ Reduced (tonnes)"})
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- STATE RANKING ----------------
    st.subheader(" Top States by CO₂ Reduction")

    state_rank = filtered.groupby("state", as_index=False)["estimated_co2_reduction_tonnes"].sum()
    state_rank = state_rank.sort_values("estimated_co2_reduction_tonnes", ascending=False).head(10)

    fig = px.bar(state_rank, x="state", y="estimated_co2_reduction_tonnes",
                 title="Top 10 States by CO₂ Reduction")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- EFFICIENCY ANALYSIS ----------------
    st.subheader(" Efficiency: CO₂ Reduced per MW")

    state_eff = filtered.groupby("state", as_index=False)["co2_per_mw"].mean()
    state_eff = state_eff.sort_values("co2_per_mw", ascending=False).head(10)

    fig = px.bar(state_eff, x="state", y="co2_per_mw",
                 title="Top States by CO₂ Reduction Efficiency")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- CLUSTERING ----------------
    st.subheader(" State Clustering (Adoption & Efficiency Patterns)")

    cluster_df = df.groupby("state").agg({
        "solar_pumps_installed": "sum",
        "solar_mw_district": "sum",
        "wind_mw": "sum",
        "bioenergy_mw": "sum",
        "co2_per_mw": "mean"
    }).reset_index()

    features = [
        "solar_pumps_installed", "solar_mw_district",
        "wind_mw", "bioenergy_mw", "co2_per_mw"
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_df[features])

    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_df["cluster"] = kmeans.fit_predict(X)

    fig = px.scatter(
        cluster_df,
        x="solar_mw_district",
        y="solar_pumps_installed",
        color="cluster",
        hover_name="state",
        title="State Clusters Based on Renewable Adoption"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- FORECASTING ----------------
    st.subheader("Forecast of Solar Pump Adoption (All India)")

    ts = df.groupby("year_month")["solar_pumps_installed"].sum()
    ts.index = pd.to_datetime(ts.index)

    model = ARIMA(ts, order=(2, 1, 2))
    fit = model.fit()

    forecast = fit.forecast(steps=24)

    future_index = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1),
                                 periods=24, freq="MS")

    forecast_df = pd.DataFrame({
        "date": future_index,
        "forecast": forecast.values
    })

    hist = ts.reset_index()
    hist.columns = ["date", "value"]

    fig = px.line(hist, x="date", y="value", title="Solar Pump Adoption Forecast")
    fig.add_scatter(x=forecast_df["date"], y=forecast_df["forecast"],
                    mode="lines", name="Forecast")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- DATA TABLE ----------------
    st.subheader("View Filtered Data")
    st.dataframe(filtered.head(500))


with predict_tab:
    st.subheader(" ML Prediction: CO₂ Reduction Estimator")

    rf_c1, rf_c2 = st.columns(2)
    with rf_c1:
        n_estimators = st.slider("Trees (n_estimators)", 50, 400, 150, step=25)
    with rf_c2:
        max_depth_opt = st.selectbox("Max depth", ["Auto", 8, 12, 15, 20, 25, 30], index=2)
    max_depth_val = None if max_depth_opt == "Auto" else int(max_depth_opt)

    ml_model, r2 = train_model(df, n_estimators=n_estimators, max_depth=max_depth_val)

    st.write(f"Model R² Score: **{r2:.4f}** (Random Forest)")

    st.markdown("Edit any feature values below and click Predict to see CO₂ estimates.")

    # Prefill editable table with reasonable defaults; users can add/remove rows
    default_row = pd.DataFrame([{
        "state": df["state"].iloc[0],
        "year": int(df["year"].max()),
        "month": 6,
        "solar_mw_district": float(df["solar_mw_district"].median()),
        "wind_mw": float(df["wind_mw"].median()),
        "bioenergy_mw": float(df["bioenergy_mw"].median()),
        "solar_pumps_installed": float(df["solar_pumps_installed"].median())
    }])

    editable_rows = st.data_editor(
        default_row,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "state": st.column_config.SelectboxColumn(
                "State",
                options=sorted(df["state"].unique().tolist()),
                help="Choose or type a state; unseen states are ignored safely."
            ),
            "year": st.column_config.NumberColumn("Year", step=1, min_value=int(df["year"].min()), max_value=2040),
            "month": st.column_config.NumberColumn("Month", step=1, min_value=1, max_value=12),
            "solar_mw_district": st.column_config.NumberColumn("Solar Capacity (MW)", format="%.2f"),
            "wind_mw": st.column_config.NumberColumn("Wind Capacity (MW)", format="%.2f"),
            "bioenergy_mw": st.column_config.NumberColumn("Bioenergy (MW)", format="%.2f"),
            "solar_pumps_installed": st.column_config.NumberColumn("Solar Pumps Installed", format="%.0f")
        }
    )

    if st.button("Predict CO₂ Reduction", type="primary"):
        if editable_rows.empty:
            st.warning("Add at least one row to predict.")
        else:
            preds = ml_model.predict(editable_rows)
            result_df = editable_rows.copy()
            result_df["predicted_co2_tonnes"] = preds
            st.success("Predictions generated.")
            st.dataframe(result_df, use_container_width=True)

    with st.expander("Feature importance (Random Forest)"):
        try:
            # Extract friendly feature names from pipeline
            feature_names = ml_model["prep"].get_feature_names_out()
            importances = ml_model["model"].feature_importances_
            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).head(12)
            fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                             title="Top Feature Importances")
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as exc:  # pragma: no cover - defensive
            st.warning(f"Could not compute feature importances: {exc}")
