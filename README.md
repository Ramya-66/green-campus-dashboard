# Green Campus Dashboard

## Overview

**Green Campus Dashboard** is a project built on the Streamlit platform to visualize, monitor, and analyze sustainability metrics for a campus environment. The system integrates real-time IoT data simulation with historical datasets to provide interactive insights and trend predictions.

## Features

- Interactive dashboards and visualizations
- Real-time data display (if connected to live data sources)
- User-friendly interface with Streamlit
- Modular design for easy expansion

## Getting Started

### Prerequisites

- Python 3.7 or higher
- [Streamlit](https://streamlit.io/) library

### Installation

Clone this repository:
```bash
git clone https://github.com/Ramya-66/green-campus-dashboard.git
cd green-campus-dashboard
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Dashboard
- To run **predefined csv data** run the Streamlit server as:
```bash
streamlit run app.py
```

- To run **live data** run the Streamlit server as:
```bash
python iot_simulator.py
```  

## Usage

- Open `http://localhost:8501` in your browser after running the app.
- Explore the dashboard features to view campus sustainability metrics.


## Contact

Created by [Ramya-66](https://github.com/Ramya-66).

---
**Green Campus Dashboard in Streamlit Platform**
