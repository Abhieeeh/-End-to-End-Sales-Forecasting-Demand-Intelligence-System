# End-to-End Sales Forecasting & Demand Intelligence System

An interactive, premium-designed Streamlit web application built to analyze sales performance, forecast future demand, detect weekly sales anomalies, and segment product sub-categories.

---

## 📁 Project Structure

* **`App/app.py`**: The main Streamlit web application containing the interactive dashboards.
* **`Notebook/analysis.ipynb`**: Jupyter Notebook containing exploratory data analysis, time-series stationarity testing, model training (SARIMA, Prophet, XGBoost), anomaly detection, and cluster analysis.
* **`Dataset/`**: Input datasets:
  * `train.csv`: Historical retail transaction sales data.
  * `vgsales.csv`: Video games sales reference dataset.
* **`Charts/`**: Exported visualizations from the models and analysis pipelines (e.g. anomaly detection plots, K-Means clustering, and forecast projections).
* **`requirements.txt`**: Package dependencies required for running the application locally and deploying to cloud platforms (such as Streamlit Cloud).

---

## 🛠️ Setup & Installation

To run the Streamlit web application locally, follow these steps:

1. **Install Dependencies**:
   Ensure you have Python installed, then run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Web Application**:
   Start the Streamlit server from the root of the project directory:
   ```bash
   streamlit run App/app.py
   ```

---

## 🖥️ Web App Features

The web application is organized into 4 interactive pages accessible via the sidebar navigation:

1. **📊 Sales Overview Dashboard**:
   * Filter historical data dynamically by **Region** and **Category**.
   * View glassmorphism-styled KPI cards for *Total Revenue*, *Total Order Count*, and *Average Ticket Size*.
   * Analyze sales trends via Yearly Sales bar charts, Monthly Trend line plots, and Category breakdowns per Region.

2. **🔮 Forecast Explorer**:
   * Forecast future sales at either the **Category** or **Region** level.
   * Adjust the forecast horizon slider (1, 2, or 3 months ahead into 2018).
   * View interactive Plotly charts showing historical sales alongside the SARIMA-predicted trend line and its 95% confidence interval band.
   * Review model validation metrics (**MAE** and **RMSE**) calculated on a 12-month holdout test set.

3. **🚨 Anomaly Report**:
   * Review the weekly sales anomalies chart generated using an **Isolation Forest** model.
   * View a tabular list of detected anomaly weeks and their corresponding weekly sales totals.

4. **🧩 Demand Segments**:
   * Inspect product sub-categories grouped via **K-Means Clustering** based on total sales value and order frequency.
   * View a demand segment table categorization (**High Demand**, **Moderate Demand**, and **Low Demand**).