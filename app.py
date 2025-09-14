# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import os

# -------------------------
# Page config + CSS (keeps your theme)
# -------------------------
st.set_page_config(page_title="Green Campus AI Dashboard", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif !important; }
    .green-box {
        background: linear-gradient(150deg, #a7ed8b, #183e08);
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }
    .stMetric > div { text-align:center; }
    .block-container { padding-top: 1rem !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="green-box">Green Campus AI Dashboard</div>', unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------
def try_parse_dates(series):
    """Try parsing a pandas Series with default (year-first) then dayfirst; return parsed series."""
    parsed_default = pd.to_datetime(series, errors='coerce', dayfirst=False)
    nonnull_default = parsed_default.notna().sum()
    frac_default = nonnull_default / max(len(series), 1)

    if frac_default >= 0.9:
        return parsed_default

    parsed_dayfirst = pd.to_datetime(series, errors='coerce', dayfirst=True)
    nonnull_df = parsed_dayfirst.notna().sum()
    frac_df = nonnull_df / max(len(series), 1)

    # choose the one with more non-null parses
    return parsed_dayfirst if frac_df > frac_default else parsed_default

def find_column(possible_names, df_columns):
    for n in possible_names:
        if n in df_columns:
            return n
    return None

def normalize_date_input(date_input, min_date, max_date):
    # Accept: single date, list/tuple of 2, numpy/pandas array, pd.Series
    try:
        if isinstance(date_input, (list, tuple, pd.Series, np.ndarray)):
            if len(date_input) >= 2:
                s, e = date_input[0], date_input[1]
            elif len(date_input) == 1:
                s = e = date_input[0]
            else:
                s, e = min_date, max_date
        else:
            s = e = date_input

        s = pd.to_datetime(s, errors='coerce')
        e = pd.to_datetime(e, errors='coerce')
        if pd.isna(s):
            s = pd.to_datetime(min_date)
        if pd.isna(e):
            e = pd.to_datetime(max_date)
        return s.date(), e.date()
    except Exception:
        return min_date, max_date

# -------------------------
# Load datasets (historical + live)
# -------------------------
hist_file = "green_campus_dataset.csv"
live_file = "live_data.csv"

datasets = []
if os.path.exists(hist_file):
    try:
        d = pd.read_csv(hist_file)
        datasets.append(d)
    except Exception as e:
        st.warning(f"Could not read {hist_file}: {e}")

if os.path.exists(live_file):
    try:
        d = pd.read_csv(live_file)
        datasets.append(d)
    except Exception as e:
        st.warning(f"Could not read {live_file}: {e}")

if not datasets:
    st.error("No data files found. Place 'live_data.csv' or 'green_campus_dataset.csv' in the app folder, or run the simulator.")
    st.stop()

# concatenate
df = pd.concat(datasets, ignore_index=True, sort=False)

# detect date-like column and parse robustly
date_col = find_column(["Date", "Timestamp", "time", "datetime"], df.columns)
if not date_col:
    st.error("No 'Date' or 'Timestamp' column found in data files.")
    st.stop()

df[date_col] = try_parse_dates(df[date_col])
df = df.dropna(subset=[date_col]).copy()
df.rename(columns={date_col: "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])  # ensure dtype

# map possible column names to canonical names
energy_col = find_column(["Energy_kWh", "Energy_Consumption", "Energy", "energy_kwh"], df.columns)
water_col  = find_column(["Water_Liters", "Water_Liter", "Water_L", "Water"], df.columns)
occ_col    = find_column(["Occupancy", "People", "Occupancy_Count", "Occupants"], df.columns)

# if missing numeric columns, create them with zeros (prevents KeyErrors)
if not energy_col:
    df["Energy_kWh"] = 0.0
else:
    df["Energy_kWh"] = pd.to_numeric(df[energy_col], errors="coerce").fillna(0.0)
if not water_col:
    df["Water_Liters"] = 0.0
else:
    df["Water_Liters"] = pd.to_numeric(df[water_col], errors="coerce").fillna(0.0)
if not occ_col:
    df["Occupancy"] = 0
else:
    df["Occupancy"] = pd.to_numeric(df[occ_col], errors="coerce").fillna(0)

# Ensure Building exists
if "Building" not in df.columns:
    st.error("No 'Building' column present in data. Add a Building column in your CSVs.")
    st.stop()

# dedupe & sort
df = df.drop_duplicates().sort_values("Date").reset_index(drop=True)

# add Month and Weekday (weekday name)
df["Month"] = df["Date"].dt.month
df["Weekday"] = df["Date"].dt.day_name()

# -------------------------
# Sidebar widgets with session_state persistence
# -------------------------
st.sidebar.header("FILTERS")

building_options = sorted(df["Building"].unique().tolist())
if "building_selected" not in st.session_state or st.session_state.building_selected not in building_options:
    st.session_state.building_selected = building_options[0] if building_options else None

building_selected = st.sidebar.selectbox(
    "Select Building",
    building_options,
    index=building_options.index(st.session_state.building_selected),
    key="building_selected"
)

# date defaults
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
if "date_range" not in st.session_state:
    default_start = (pd.to_datetime(max_date) - pd.Timedelta(days=30)).date()
    if default_start < min_date:
        default_start = min_date
    st.session_state.date_range = (default_start, max_date)

# show date_input (list so user gets range widget)
date_input_widget = st.sidebar.date_input(
    "Select Date Range",
    value=list(st.session_state.date_range),
    min_value=min_date,
    max_value=max_date,
    key="date_input_widget"
)

# normalize what user returned
start_date, end_date = normalize_date_input(date_input_widget, min_date, max_date)
st.session_state.date_range = (start_date, end_date)

# quick buttons
st.sidebar.markdown("---")
if st.sidebar.button("Past 7 days"):
    st.session_state.date_range = ((pd.to_datetime(max_date) - pd.Timedelta(days=7)).date(), max_date)
if st.sidebar.button("Past 30 days"):
    st.session_state.date_range = ((pd.to_datetime(max_date) - pd.Timedelta(days=30)).date(), max_date)
if st.sidebar.button("This month"):
    st.session_state.date_range = (pd.to_datetime(max_date).replace(day=1).date(), max_date)

# use final values
start_date, end_date = st.session_state.date_range
start_date = pd.to_datetime(start_date).date()
end_date   = pd.to_datetime(end_date).date()

# manual refresh button (safe)
if "manual_refresh" not in st.session_state:
    st.session_state.manual_refresh = False
if st.sidebar.button("🔄 Refresh Data (manual)"):
    st.session_state.manual_refresh = not st.session_state.manual_refresh

# try to use streamlit_autorefresh if installed (doesn't reset session_state)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=20_000, key="auto_refresh")
except Exception:
    pass

# -------------------------
# Time-window filter
# -------------------------
mask_time = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
df_window = df.loc[mask_time].copy()

if df_window.empty:
    st.warning("⚠️ No data in the selected date range. Widen the range or check the simulator.")
    st.stop()

# filter by building (final)
df_filtered = df_window[df_window["Building"] == building_selected].sort_values("Date")
if df_filtered.empty:
    st.warning("⚠️ No records for this building in the selected date range.")
    st.stop()

# -------------------------
# Prepare ML dataset safely
# -------------------------
# We'll train on df_window (time-window) to keep it relevant
train_df = df_window.copy()
# ensure Weekday exists (string names)
train_df["Weekday"] = train_df["Date"].dt.day_name()

# one-hot encode building + weekday
df_model = pd.get_dummies(train_df, columns=["Building", "Weekday"], drop_first=True)

# ensure numeric security for model inputs
for c in ["Energy_kWh", "Water_Liters", "Occupancy"]:
    if c not in df_model.columns:
        df_model[c] = 0
    df_model[c] = pd.to_numeric(df_model[c], errors="coerce").fillna(0)

# features and target
X = df_model.drop(columns=["Date", "Energy_kWh"], errors="ignore")
# keep numeric only
X = X.select_dtypes(include=[np.number]).fillna(0)
y = df_model["Energy_kWh"].fillna(0)

r2, mae = 0.0, 0.0
trained = False
model = None
if len(X) >= 2 and X.shape[1] >= 1 and y.nunique() > 1:
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        trained = True
    except Exception as e:
        st.warning(f"ML training failed: {e}")
        trained = False
else:
    # not enough data to train reliably; we'll fallback to actual values for predictions
    trained = False

# produce predictions for df_filtered
if trained and model is not None:
    # create prediction features aligned with X columns
    df_pred = df_filtered.copy()
    df_pred["Weekday"] = df_pred["Date"].dt.day_name()
    df_pred = pd.get_dummies(df_pred, columns=["Building", "Weekday"], drop_first=True)
    Xp = df_pred.reindex(columns=X.columns, fill_value=0).select_dtypes(include=[np.number]).fillna(0)
    try:
        df_filtered["Predicted_Energy"] = model.predict(Xp)
    except Exception:
        df_filtered["Predicted_Energy"] = df_filtered["Energy_kWh"]
else:
    df_filtered["Predicted_Energy"] = df_filtered["Energy_kWh"]

# -------------------------
# Sidebar model info & download
# -------------------------
st.sidebar.markdown(f"**Model R² Score:** `{r2:.2f}`")
st.sidebar.markdown(f"**Mean Absolute Error:** `{mae:.2f}`")
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download Filtered Data", data=csv, file_name='filtered_data.csv', mime='text/csv')

# -------------------------
# KPIs
# -------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Avg. Energy (kWh)", f"{df_filtered['Energy_kWh'].mean():.2f}")
col2.metric("Avg. Water Usage (L)", f"{df_filtered['Water_Liters'].mean():.2f}")
col3.metric("Avg. Occupancy", f"{df_filtered['Occupancy'].mean():.0f}")

# -------------------------
# Charts (colors preserved)
# -------------------------
tab1, tab2, tab3 = st.tabs(["Energy & Prediction", "Water Usage", "Occupancy"])

with tab1:
    fig1 = px.line(df_filtered, x="Date", y=["Energy_kWh", "Predicted_Energy"],
                   labels={"value": "Energy (kWh)", "variable": "Type"},
                   color_discrete_map={"Energy_kWh": "#FFFDD0", "Predicted_Energy": "#8B4513"},
                   title="Energy Usage with Predictions")
    fig1.update_layout(title_font_size=18, font=dict(family="Segoe UI", size=14), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.line(df_filtered, x="Date", y="Water_Liters",
                   title="Water Consumption",
                   labels={"Water_Liters": "Water (L)"},
                   color_discrete_sequence=['teal'])
    fig2.update_layout(title_font_size=18, font=dict(family="Segoe UI", size=14), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.bar(df_filtered, x="Date", y="Occupancy",
                  title="Occupancy Count Over Time",
                  labels={"Occupancy": "People Count"},
                  color_discrete_sequence=["#8C8582"])
    fig3.update_layout(title_font_size=18, font=dict(family="Segoe UI", size=14), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Monthly summary + raw data
# -------------------------
st.subheader("MONTHLY SUMMARY")
monthly_summary = df_filtered.groupby(df_filtered["Date"].dt.month).agg({
    "Energy_kWh": "mean", "Water_Liters": "mean", "Occupancy": "mean"
}).rename_axis("Month").round(2)
st.dataframe(monthly_summary)

with st.expander("View Raw Data"):
    st.dataframe(df_filtered)
