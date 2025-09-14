**Green Campus Dashboard**

A Streamlit-based dashboard to monitor and predict sustainability metrics (energy, water,occupancy, etc.) for a green campus initiative. The system integrates real-time IoT data simulation with historical datasets to provide interactive insights and trend predictions.

**File Structure**

    green-campus-dashboard
  
      * app.py (Main Streamlit app production version)            
    
      *  app-original.py (Backup or reference version of app.py)
    
      *  iot_simulator.py (Script to simulate IoT data and update CSV)
    
      *  live_data.csv (Real-time or simulated input data file)
    
      *  green_campus_dataset.csv (Historical dataset for ML/trends)
    
      *  requirements.txt (Python dependencies for Streamlit Cloud)

**Running Locally**

     # Run the simulator to continuously generate data in live_data.csv
        python iot_simulator.py
        
     # Run Streamlit Dashboar
        streamlit run app.py

**Dashboard** (*app.py*)

   - Loads and processes data using pandas.
   - Visualizes trends with plotly and matplotlib.
   - Runs ML predictions (e.g., LinearRegression from scikit-learn).
   - Displays interactive charts and KPIs via Streamlit widgets.
     
**Deployment**
   - All files are version-controlled in GitHub.
   - Streamlit Cloud automatically redeploys when changes are pushed.
   - Users access the live dashboard via a public URL.


  
