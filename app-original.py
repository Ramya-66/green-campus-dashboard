import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Page configuration
st.set_page_config(page_title="Green Campus AI Dashboard", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif !important;
    }

    .green-box {
        background: linear-gradient(150deg, #a7ed8b, #183e08);
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.4);
    }

    .stMetric > div {
        text-align: center;
    }

    .block-container {
        padding-top: 2rem !important;
    }

    .stDataFrame th, .stDataFrame td {
        font-family: 'Segoe UI', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="green-box">Green Campus AI Dashboard</div>', unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("green_campus_dataset.csv", parse_dates=["Date"])

# Feature engineering
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.day_name()

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Building', 'Weekday'], drop_first=True)

# ML Model: Predict Energy_kWh
X = df_encoded.drop(columns=['Date', 'Energy_kWh'])
y = df_encoded['Energy_kWh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
r2 = r2_score(y_test, model.predict(X_test))
mae = mean_absolute_error(y_test, model.predict(X_test))

# Prediction column
df['Predicted_Energy'] = model.predict(df_encoded.drop(columns=['Date', 'Energy_kWh']))

# Sidebar filters
st.sidebar.header("FILTERS")
building_selected = st.sidebar.selectbox("Select Building", df['Building'].unique())

min_date = df['Date'].min()
max_date = df['Date'].max()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Show model evaluation
st.sidebar.markdown(f"**Model R² Score:** `{r2:.2f}`")
st.sidebar.markdown(f"**Mean Absolute Error:** `{mae:.2f}`")

# Apply filters
df_filtered = df[
    (df['Building'] == building_selected) &
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# Summary Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Avg. Energy (kWh)", f"{df_filtered['Energy_kWh'].mean():.2f}")
col2.metric("Avg. Water Usage (L)", f"{df_filtered['Water_Liters'].mean():.2f}")
col3.metric("Avg. Occupancy", f"{df_filtered['Occupancy'].mean():.0f}")

# Tabs for charts
tab1, tab2, tab3 = st.tabs(["Energy & Prediction", "Water Usage", "Occupancy"])

with tab1:
    fig1 = px.line(df_filtered, x='Date', y=['Energy_kWh', 'Predicted_Energy'],
                   labels={'value': 'Energy (kWh)', 'variable': 'Type'},
                   color_discrete_map={'Energy_kWh': "#FFFDD0", 'Predicted_Energy': "#FF7F11"},
                   title='Energy Usage with Predictions')
    fig1.update_layout(title_font_size=18, font=dict(family="Segoe UI", size=14))
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.line(df_filtered, x='Date', y='Water_Liters',
                   title='Water Consumption',
                   labels={'Water_Liters': 'Water (L)'},
                   color_discrete_sequence=['teal'])
    fig2.update_layout(title_font_size=18, font=dict(family="Segoe UI", size=14))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.bar(df_filtered, x='Date', y='Occupancy',
                  title='Occupancy Count Over Time',
                  labels={'Occupancy': 'People Count'},
                  color_discrete_sequence=["#e67ead"])
    fig3.update_layout(title_font_size=18, font=dict(family="Segoe UI", size=14))
    st.plotly_chart(fig3, use_container_width=True)

# Monthly Summary Table
st.subheader("MONTHLY SUMMARY")
monthly_summary = df[df['Building'] == building_selected].groupby('Month').agg({
    'Energy_kWh': 'mean',
    'Water_Liters': 'mean',
    'Occupancy': 'mean'
}).round(2)
st.dataframe(monthly_summary)

# Download button
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Data", data=csv, file_name='filtered_data.csv', mime='text/csv')

# Raw Data
with st.expander("View Raw Data"):
    st.dataframe(df_filtered)
