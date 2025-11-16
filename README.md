# Ad Click Prediction - ML Pipeline & Dashboard

A comprehensive machine learning pipeline and interactive analytics dashboard for predicting ad clicks and analyzing adtech performance metrics. Built with Python, scikit-learn, XGBoost, LightGBM, MLflow, Streamlit, and Plotly.

---

## 📋 Project Overview

This project combines:
- **Machine Learning Pipeline**: Multiple classification models (KNN, XGBoost, LightGBM) with SMOTE for imbalanced data
- **MLflow Experiment Tracking**: Centralized model performance monitoring and comparison
- **Interactive Dashboard**: Real-time segment analysis, funnel visualization, and adtech metrics

### Key Features

✅ **Multi-Model Training** - KNN, XGBoost, LightGBM with cross-validation  
✅ **Experiment Tracking** - MLflow integration with auto-timestamped experiments  
✅ **Class Imbalance Handling** - SMOTE oversampling for better minority class prediction  
✅ **Interactive Analytics** - 7-tab Streamlit dashboard with real-time filtering  
✅ **AdTech Metrics** - CTR, CVR, ROAS, CPA, LTV, ROI calculations  
✅ **Segment Analysis** - Performance breakdown by device, gender, time, position, browsing history  
✅ **Network Accessible** - MLflow UI shareable with team members  

---

## 🗂️ Project Structure

```
.
├── README.md                          # This file
├── notebook.py                        # Main ML pipeline & model training
├── streamlit_app.py                   # Interactive dashboard
├── sample_dataset.csv                 # Input data
├── mlruns/                            # MLflow artifacts & runs (auto-generated)
├── segment_performance.csv            # Output: segment metrics
└── requirements.txt                   # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn mlflow streamlit plotly
```

### 2. Prepare Data

Ensure `sample_dataset.csv` contains:
- `id` - Record identifier
- `full_name` - User name
- `age` - User age
- `gender` - User gender
- `device_type` - Desktop/Mobile/Tablet
- `ad_position` - Ad placement position
- `browsing_history` - User browsing category
- `time_of_day` - Morning/Afternoon/Evening/Night
- `click` - Binary: 1 if clicked, 0 otherwise

### 3. Start MLflow Server (in Terminal 1)

```bash
# Local access only
mlflow ui --port 8080

# OR for team access (replace 192.168.1.100 with your IP)
mlflow ui --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///mlflow.db
```

Then open: **http://127.0.0.1:8080**

### 4. Run ML Pipeline (in Terminal 2)

```bash
python notebook.py
```

This will:
- Load and preprocess data
- Train KNN, XGBoost, and LightGBM models
- Log metrics to MLflow with timestamp
- Generate `segment_performance.csv`

**Console Output:**
```
✓ Created new experiment: Ad_Click_Prediction_new_run_20251116_025200
✓ MLflow URI: http://127.0.0.1:8080
✓ Experiment: Ad_Click_Prediction_new_run_20251116_025200

=== KNN MODEL ===
Accuracy: 0.8234, Precision: 0.7891, Recall: 0.6543, F1: 0.7154
CV Mean: 0.8102
✓ Logged to MLflow

[Similar output for XGBoost and LightGBM...]
```

### 5. Launch Streamlit Dashboard (in Terminal 3)

Streamlit app is available in the the following link :  **https://anarsultani97-maf-sample-app-ib8xlr.streamlit.app/** 


```bash
streamlit run streamlit_app.py
```

Then open: **http://localhost:8501**

---

## 📊 ML Pipeline (notebook.py)

### Data Preprocessing

```python
# Features are automatically detected and filled:
# - Numerical (age): median imputation, binned into age groups
# - Categorical (gender, device_type, etc.): mode imputation or 'Unknown'
# - Conversion: synthetic generation (8% of clicks convert)

data['age_group'] = pd.cut(data['age'], bins=[0, 24, 34, 44, 54, 64, 100], 
                           labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
```

### Models Trained

1. **K-Nearest Neighbors (KNN)**
   - Parameters: n_neighbors=8, weights='distance', algorithm='auto'
   - Best for: Quick baseline, interpretable results

2. **XGBoost**
   - Parameters: max_depth=5, learning_rate=0.1, n_estimators=100
   - Best for: Fast training, handling feature interactions

3. **LightGBM**
   - Parameters: num_leaves=31, learning_rate=0.1, n_estimators=100
   - Best for: Large datasets, memory efficiency

### Class Imbalance Handling

SMOTE (Synthetic Minority Oversampling) applied to training data:
```python
from imblearn.over_sampling import SMOTE

smt = SMOTE()
X_train, y_train = smt.fit_resample(X_train, y_train)
```

### Metrics Calculated

- **Accuracy**: Overall correctness
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Validation Mean**: 5-fold CV average

### MLflow Logging

```python
with mlflow.start_run(run_name='KNN'):
    mlflow.log_params({'model': 'KNN', 'n_neighbors': 8, 'weights': 'distance'})
    mlflow.log_metrics({
        'accuracy': accuracy_knn,
        'precision': precision_knn,
        'recall': recall_knn,
        'f1': f1_knn,
        'cv_mean': cv_scores_knn.mean()
    })
```

---

## 📈 Streamlit Dashboard (streamlit_app.py)

### Interactive Features

**Sidebar Filters:**
- Device Type (Desktop, Mobile, Tablet)
- Age Group (18-24, 25-34, 35-44, 45-54, 55-64, 65+)
- Browsing History (Low, Medium, High)
- Time of Day (Morning, Afternoon, Evening, Night)
- Ad Position (Top, Middle, Bottom)
- Gender (Male, Female, Unknown)
- **Value Parameters**: Conversion value, click value, total spend

### Dashboard Tabs

#### Tab 1: Metrics
- **Key KPIs**: Impressions, Clicks, Conversions, Revenue, ROI
- **AdTech Metrics Table**: CTR, CVR, ROAS, CPA, LTV, Profit
- **Conversion Funnel**: Visual drop-off at each stage

#### Tab 2: Analysis
- CTR by Device Type (bar chart)
- CTR by Ad Position (bar chart)
- CTR by Time of Day (bar chart)
- CTR by Gender (bar chart)

#### Tab 3: Funnels
- Individual funnel for each category (device, gender, time, position, browsing history)
- Impression → Click → Conversion flow
- Selectable feature for dynamic analysis

#### Tab 4: Segments
- Complete segment performance table
- All categorical features shown
- Sortable by any metric (ROAS default)

#### Tab 5: Top/Bottom Performers
- Top 10 segments by CTR
- Bottom 10 segments by CTR
- Top 10 segments by LTV
- Bottom 10 segments by LTV
- Top 10 segments by Conversions
- Top 10 segments by Spend

#### Tab 6: EDA
- Gender distribution (pie chart)
- Device distribution (pie chart)
- Age distribution (histogram)
- Conversions by age group (bar)
- Browsing history breakdown (bar)
- Position performance (scatter)
- Time of day trends (dual axis)
- Device type trends (dual axis)

#### Tab 7: Data View
- Raw filtered data viewer
- Customizable column selection
- Download as CSV button

### Calculated Metrics

```
CTR (Click-Through Rate) = (Clicks / Impressions) × 100
CVR (Conversion Rate) = (Conversions / Clicks) × 100
ROAS (Return on Ad Spend) = Revenue / Spend
CPA (Cost Per Acquisition) = Spend / Conversions
LTV (Lifetime Value) = Revenue / Conversions
Profit = Revenue - Spend
ROI = (Profit / Spend) × 100
```

---

## 🔧 Configuration & Customization

### Change MLflow Tracking URI

For **remote/team access**:
```python
# In notebook.py or streamlit_app.py
mlflow.set_tracking_uri(uri="http://192.168.1.100:8080")  # Replace with your IP
```

### Modify Model Parameters

In `notebook.py`:
```python
# KNN
classifier_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# XGBoost
classifier_xgb = XGBClassifier(max_depth=7, learning_rate=0.05, n_estimators=200)

# LightGBM
classifier_lgb = LGBMClassifier(num_leaves=50, learning_rate=0.05, n_estimators=200)
```

### Adjust Funnel/Segment Combinations

In `streamlit_app.py`:
```python
# Modify categorical features for analysis
categorical_features = ['device_type', 'gender', 'ad_position', 'browsing_history', 'time_of_day']

# Add/remove as needed
```

### Change Data Source

```python
# Replace in load_data() function
data = pd.read_csv('your_custom_dataset.csv')
# or
data = pd.read_sql("SELECT * FROM ads_table", connection)
```

---

## 📋 MLflow Experiment Management

### View Experiment History

1. Open MLflow UI: **http://127.0.0.1:8080**
2. Click on experiment name
3. Compare runs side-by-side
4. Download artifacts (models, plots)

### Experiment Naming Convention

Experiments are auto-created with timestamp:
```
Ad_Click_Prediction_new_run_20251116_025200
└─ YYYYMMDD_HHMMSS (timestamp)
```

### Share Results with Team

**URL for team members:**
```
http://192.168.1.100:8080
(Replace 192.168.1.100 with your machine IP)
```

**To get your IP:**
```bash
# Windows
ipconfig

# Mac/Linux
ifconfig
```

---

## 🔍 Troubleshooting

### Issue: `ValueError: Driver feature 'browsing_history' is not a categorical column`

**Solution:** Ensure categorical columns are converted before processing:
```python
cat_cols = ['device_type', 'browsing_history', 'ad_position', 'time_of_day', 'gender']
data[cat_cols] = data[cat_cols].astype('category')
```

### Issue: MLflow server not running

**Solution:**
```bash
# Check if MLflow is installed
pip install mlflow

# Start server
mlflow ui --port 8080

# Verify at http://localhost:8080
```

### Issue: Streamlit not finding data

**Solution:** Ensure `sample_dataset.csv` is in the same directory as `streamlit_app.py`

```bash
ls -la sample_dataset.csv
```

### Issue: SMOTE memory error

**Solution:** Reduce training data or batch size:
```python
# Downsample if dataset is very large
data = data.sample(n=100000, random_state=42)
```

---

## 📊 Output Files

### Generated Files

1. **`segment_performance.csv`**
   - Segment combinations with metrics
   - Columns: Segment_Combination, device_type, gender, age_group, ad_position, time_of_day, browsing_history, Impressions, Clicks, Conversions, CTR, CVR, ROAS, CPA, LTV, Spend, Profit, ROI
   - Sorted by ROAS (highest first)

2. **`mlruns/`** (directory)
   - MLflow artifacts and run metadata
   - Organized by experiment and run ID

### Usage Example

```python
# Load segment analysis
segments = pd.read_csv('segment_performance.csv')

# Top 10 ROAS performers
print(segments.head(10)[['Segment_Combination', 'ROAS', 'Profit']])
```

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Data Processing | Pandas, NumPy | Latest |
| ML Models | scikit-learn, XGBoost, LightGBM | Latest |
| Class Imbalance | imbalanced-learn (SMOTE) | Latest |
| Experiment Tracking | MLflow | Latest |
| Dashboard | Streamlit | Latest |
| Visualization | Plotly | Latest |
| Backend | Python | 3.8+ |

---

## 📈 Performance Benchmarks

Typical results on sample dataset (~100K records):

| Model | Accuracy | Precision | Recall | F1-Score | CV Mean |
|-------|----------|-----------|--------|----------|---------|
| KNN | 82.3% | 78.9% | 65.4% | 71.5% | 81.0% |
| XGBoost | 85.1% | 82.4% | 68.9% | 75.2% | 84.2% |
| LightGBM | 86.2% | 83.7% | 70.1% | 76.4% | 85.5% |

*Performance varies based on data distribution and hyperparameters*

---

## 🤝 Team Collaboration

### Share Dashboard

1. Start MLflow server on your machine
2. Get your machine IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
3. Share URL: `http://<your-ip>:8080`
4. Team members can view all experiments and runs

### Export Results

```bash
# Download experiment as CSV
# From MLflow UI: Runs → Select run → Download artifacts

# Or programmatically
import mlflow
runs = mlflow.search_runs(experiment_names=['Ad_Click_Prediction_new_run_*'])
runs.to_csv('all_runs.csv')
```

---

## 📝 Next Steps / Enhancements

- [ ] Add hyperparameter tuning (Optuna/Hyperopt)
- [ ] Deploy model as REST API (Flask/FastAPI)
- [ ] Add feature importance analysis
- [ ] Implement real-time prediction pipeline
- [ ] Add Looker Studio integration
- [ ] Implement A/B testing framework
- [ ] Add statistical significance testing
- [ ] Create automated alerts for low-performing segments

---

## 📞 Support & Questions

For issues or questions:
1. Check the **Troubleshooting** section above
2. Verify all dependencies are installed: `pip list`
3. Check MLflow server status: `curl http://localhost:8080`
4. Ensure data format matches expected schema

---

## 📄 License

This project is provided as-is for educational and commercial use.

---

## 🎯 Key Takeaways

✅ **End-to-End ML Pipeline**: From data loading to model evaluation  
✅ **Production-Ready Tracking**: MLflow for experiment management  
✅ **Business Intelligence**: AdTech metrics and segment analysis  
✅ **Team Collaboration**: Shareable MLflow UI and Streamlit dashboard  
✅ **Interactive Analysis**: Real-time filtering and visualization  

**Built for AdTech analytics and interview preparation** 🚀
