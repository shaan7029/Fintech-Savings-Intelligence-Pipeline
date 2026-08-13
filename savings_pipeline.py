# ============================================================
# FINTECH SAVINGS INTELLIGENCE PIPELINE
# Author   : Santanu Sarkar
# Dataset  : Customer Financial Profiles — India
#            20,000 transactions | 4,941 clients | 15 cities
# Purpose  : Automated customer financial health analysis,
#            savings capacity scoring, credit segmentation,
#            and KPI reporting for fintech savings platforms
# Tools    : Python -> Excel (10 sheets) -> Power BI
# Schedule : Runs daily at 8 AM via Windows Task Scheduler
# ============================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib 
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
 
# -- Configuration ------------------------------------------
INPUT_FILE  = "D:/Jar_Financial_Intelligence/data/raw/Customer_financial_profiles.csv"
OUTPUT_FILE = "D:/Jar_Financial_Intelligence/data/processed/savings_intelligence_report.xlsx"
LOG_FILE    = "D:/Jar_Financial_Intelligence/pipeline_log.txt"
TODAY       = datetime.today()
 
# -- Logging setup -------------------------------------------
logging.basicConfig(
    filename = LOG_FILE,
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s"
)
logging.info("=" * 60)
logging.info("FINTECH SAVINGS INTELLIGENCE PIPELINE STARTED")
 
 
# ----------------------------------------------------------
# SECTION 2: DATA INGESTION
# ----------------------------------------------------------
def load_data(path):
    """Load raw CSV and parse date column."""
    try:
        df = pd.read_csv(path, parse_dates=['date'])
        logging.info(
            f"Data loaded: {df.shape[0]:,} rows, "
            f"{df.shape[1]} columns"
        )
        print(f"Loaded: {df.shape[0]:,} transactions "
              f"from {df['client_id'].nunique():,} clients")
        return df
    except FileNotFoundError:
        logging.error(f"FILE NOT FOUND: {path}")
        print(f"File not found: {path}")
        raise
 
 
# ----------------------------------------------------------
# SECTION 3: DATA CLEANING AND VALIDATION
# ----------------------------------------------------------
def clean_and_validate(df):
    """Clean the dataset and run data quality checks."""
    original_count = len(df)
    issues = []
 
    dupes = df.duplicated(subset='transaction_id').sum()
    if dupes > 0:
        df.drop_duplicates(subset='transaction_id', inplace=True)
        issues.append(f"REMOVED {dupes} duplicate transaction IDs")
 
    neg_amounts = (df['amount'] <= 0).sum()
    if neg_amounts > 0:
        df = df[df['amount'] > 0]
        issues.append(f"REMOVED {neg_amounts} zero/negative amounts")
 
    bad_scores = (
        (df['credit_score'] < 300) |
        (df['credit_score'] > 900)
    ).sum()
    if bad_scores > 0:
        issues.append(f"FLAGGED {bad_scores} invalid credit scores")
 
    future_dates = (df['date'] > TODAY).sum()
    if future_dates > 0:
        issues.append(f"FOUND {future_dates} future-dated transactions")
 
    total_nulls = df.isnull().sum().sum()
    issues.append(
        f"NULL CHECK: {total_nulls} total nulls "
        f"({'CLEAN' if total_nulls == 0 else 'NEEDS ATTENTION'})"
    )
 
    for issue in issues:
        logging.info(f"DATA QUALITY | {issue}")
 
    removed = original_count - len(df)
    logging.info(
        f"Cleaning complete. Removed {removed} rows. "
        f"{len(df):,} records remain."
    )
    print(f"Validation done — {len(df):,} clean records")
    return df
 
 
# ----------------------------------------------------------
# SECTION 4: FEATURE ENGINEERING
# ----------------------------------------------------------
def engineer_features(df):
    """Create new business-relevant features."""
 
    df['year']        = df['date'].dt.year
    df['month']       = df['date'].dt.month
    df['month_name']  = df['date'].dt.month_name()
    df['quarter']     = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend']  = df['date'].dt.dayofweek >= 5
 
    df['age_group'] = pd.cut(
        df['current_age'],
        bins   = [18, 25, 35, 45, 55, 65, 100],
        labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    )
 
    df['income_segment'] = pd.cut(
        df['yearly_income'],
        bins   = [0, 300000, 600000, 1000000, 1500000, 9999999],
        labels = ['Low (<3L)', 'Lower-Mid (3-6L)', 'Mid (6-10L)',
                  'Upper-Mid (10-15L)', 'High (>15L)']
    )
 
    df['credit_category'] = pd.cut(
        df['credit_score'],
        bins   = [300, 580, 670, 740, 800, 901],
        labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    )
 
    df['debt_to_income'] = (
        df['total_debt'] / df['yearly_income']
    ).round(4)
 
    credit_norm = (df['credit_score'] - 300) / 600
    dti_norm    = 1 - df['debt_to_income'].clip(0, 1)
    income_norm = (
        np.log1p(df['yearly_income']) /
        np.log1p(df['yearly_income'].max())
    )
    df['financial_health_score'] = (
        credit_norm * 50 +
        dti_norm    * 30 +
        income_norm * 20
    ).round(2)
 
    df['health_tier'] = pd.cut(
        df['financial_health_score'],
        bins   = [0, 40, 60, 75, 101],
        labels = ['At Risk', 'Needs Attention', 'Stable', 'Healthy']
    )
 
    df['monthly_income']   = df['yearly_income'] / 12
    df['monthly_debt_emi'] = df['total_debt'] * 0.02
    df['savings_capacity'] = (
        df['monthly_income'] - df['amount'] - df['monthly_debt_emi']
    ).round(2)
    df['savings_capacity'] = df['savings_capacity'].clip(lower=0)
 
    df['savings_segment'] = pd.cut(
        df['savings_capacity'],
        bins   = [0, 5000, 15000, 30000, 9999999],
        labels = ['Low (<5K/mo)', 'Medium (5-15K)',
                  'High (15-30K)', 'Premium (>30K)']
    )
 
    df['payment_method'] = df['use_chip'].map(
        {'Yes': 'Chip/Digital', 'No': 'Swipe/Manual'}
    )
 
    high_risk = (df['health_tier'] == 'At Risk').sum()
    healthy   = (df['health_tier'] == 'Healthy').sum()
    logging.info(
        f"Features engineered. Healthy: {healthy:,} | At Risk: {high_risk:,}"
    )
    print("Features engineered")
    return df
 
 
# ----------------------------------------------------------
# SECTION 5: KPI CALCULATION
# ----------------------------------------------------------
def calculate_kpis(df):
    """Calculate all top-level business KPIs."""
    total_txn     = len(df)
    total_clients = df['client_id'].nunique()
    total_value   = df['amount'].sum()
    avg_income    = df.groupby('client_id')['yearly_income'].first().mean()
    healthy_pct   = (df['health_tier'] == 'Healthy').mean() * 100
    at_risk_pct   = (df['health_tier'] == 'At Risk').mean() * 100
    avg_health    = df['financial_health_score'].mean()
    avg_savings   = df['savings_capacity'].mean()
    digital_pct   = (df['payment_method'] == 'Chip/Digital').mean() * 100
    high_credit   = (df['credit_score'] >= 740).mean() * 100
 
    kpis = {
        'Total Transactions':           f"{total_txn:,}",
        'Total Transaction Value (Rs)': f"Rs {total_value:,.0f}",
        'Avg Transaction Value (Rs)':   f"Rs {df['amount'].mean():,.0f}",
        'Unique Clients':               f"{total_clients:,}",
        'Avg Yearly Income (Rs)':       f"Rs {avg_income:,.0f}",
        'Avg Financial Health Score':   f"{avg_health:.1f} / 100",
        'Healthy Clients %':            f"{healthy_pct:.1f}%",
        'Early Warning Clients':        f"{int(df[df['financial_stress_flag']==True]['client_id'].nunique()):,}",
        'Avg Monthly Savings Capacity': f"Rs {avg_savings:,.0f}",
        'Digital Payment Adoption %':   f"{digital_pct:.1f}%",
        'High Credit Score Users %':    f"{high_credit:.1f}%",
        'Cities Covered':               f"{df['merchant_city'].nunique()}",
        'States Covered':               f"{df['merchant_state'].nunique()}",
        'Report Generated':             TODAY.strftime('%Y-%m-%d %H:%M')
    }
 
    kpi_df = pd.DataFrame(list(kpis.items()), columns=['KPI', 'Value'])
    logging.info(
        f"KPIs calculated | Healthy: {healthy_pct:.1f}% | "
        f"At Risk: {at_risk_pct:.1f}%"
    )
    print("KPIs calculated")
    return kpi_df
 
 
# ----------------------------------------------------------
# SECTION 6: SUMMARY TABLES
# ----------------------------------------------------------
def build_summary_tables(df):
    """Build all aggregation tables for Power BI sheets."""
 
    city_summary = df.groupby('merchant_city').agg(
        Transactions     = ('transaction_id', 'count'),
        Total_Value      = ('amount', 'sum'),
        Avg_Txn_Value    = ('amount', 'mean'),
        Avg_Health_Score = ('financial_health_score', 'mean'),
        Avg_Income       = ('yearly_income', 'mean'),
        Avg_Credit_Score = ('credit_score', 'mean'),
        Unique_Clients   = ('client_id', 'nunique')
    ).reset_index().round(2)
    city_summary.sort_values('Total_Value', ascending=False, inplace=True)
 
    income_seg = df.groupby('income_segment', observed=True).agg(
        Clients          = ('client_id', 'nunique'),
        Avg_Health_Score = ('financial_health_score', 'mean'),
        Avg_Savings_Cap  = ('savings_capacity', 'mean'),
        Avg_Txn_Value    = ('amount', 'mean'),
        Avg_Credit_Score = ('credit_score', 'mean')
    ).reset_index().round(2)
 
    credit_seg = df.groupby('credit_category', observed=True).agg(
        Count           = ('client_id', 'nunique'),
        Avg_Income      = ('yearly_income', 'mean'),
        Avg_Debt        = ('total_debt', 'mean'),
        Avg_Savings_Cap = ('savings_capacity', 'mean'),
        Avg_Txn_Value   = ('amount', 'mean')
    ).reset_index().round(2)
 
    monthly = df.groupby(
        [df['date'].dt.year.rename('year'),
         df['date'].dt.month.rename('month')]
    ).agg(
        Transactions   = ('transaction_id', 'count'),
        Total_Value    = ('amount', 'sum'),
        Avg_Value      = ('amount', 'mean'),
        Unique_Clients = ('client_id', 'nunique')
    ).reset_index().round(2)
    monthly.sort_values(['year', 'month'], inplace=True)
 
    age_analysis = df.groupby('age_group', observed=True).agg(
        Clients          = ('client_id', 'nunique'),
        Avg_Txn_Value    = ('amount', 'mean'),
        Avg_Income       = ('yearly_income', 'mean'),
        Avg_Health_Score = ('financial_health_score', 'mean'),
        Avg_Savings_Cap  = ('savings_capacity', 'mean')
    ).reset_index().round(2)
 
    health_dist = df.groupby('health_tier', observed=True).agg(
        Count           = ('client_id', 'nunique'),
        Avg_Income      = ('yearly_income', 'mean'),
        Avg_Credit      = ('credit_score', 'mean'),
        Avg_Debt        = ('total_debt', 'mean'),
        Avg_Savings_Cap = ('savings_capacity', 'mean')
    ).reset_index().round(2)
 
    savings_seg = df.groupby('savings_segment', observed=True).agg(
        Count      = ('client_id', 'nunique'),
        Avg_Income = ('yearly_income', 'mean'),
        Avg_Health = ('financial_health_score', 'mean'),
        Avg_Txn    = ('amount', 'mean')
    ).reset_index().round(2)
 
    print("Summary tables built")
    return (city_summary, income_seg, credit_seg,
            monthly, age_analysis, health_dist, savings_seg)
 
 
# ----------------------------------------------------------
# SECTION 7: ANOMALY DETECTION
# ----------------------------------------------------------
def detect_anomalies(df):
    """Flag financially anomalous clients."""
 
    Q1    = df['amount'].quantile(0.25)
    Q3    = df['amount'].quantile(0.75)
    IQR   = Q3 - Q1
    upper = Q3 + 1.5 * IQR
 
    df['txn_anomaly'] = np.where(
        df['amount'] > upper, 'Unusually Large', 'Normal'
    )
 
    df['financial_stress_flag'] = (
        (df['debt_to_income'] > 0.4) &
        (df['credit_score']   < 670) &
        (df['yearly_income']  < 400000)
    )
 
    alert_clients = df[
        df['financial_stress_flag'] == True
    ].groupby('client_id').agg(
        Age            = ('current_age', 'first'),
        City           = ('merchant_city', 'first'),
        Yearly_Income  = ('yearly_income', 'first'),
        Credit_Score   = ('credit_score', 'first'),
        Total_Debt     = ('total_debt', 'first'),
        Debt_to_Income = ('debt_to_income', 'first'),
        Health_Score   = ('financial_health_score', 'mean'),
        Health_Tier    = ('health_tier', 'first')
    ).reset_index().round(2)
    alert_clients.sort_values('Health_Score', inplace=True)
 
    anomaly_count  = (df['txn_anomaly'] != 'Normal').sum()
    stressed_count = df['financial_stress_flag'].sum()
    logging.info(
        f"Anomalies: {anomaly_count} large txns | "
        f"{stressed_count} financially stressed clients"
    )
    print(f"Anomalies flagged: {anomaly_count:,} large transactions | "
          f"{stressed_count:,} stressed clients")
    return df, alert_clients
 
 
# ----------------------------------------------------------
# SECTION 8: EXPORT TO EXCEL AND MASTER RUNNER
# ----------------------------------------------------------
def export_to_excel(kpi_df, city_df, income_df, credit_df,
                    monthly_df, age_df, health_df, savings_df,
                    alerts_df, churn_df, clean_df, output_path):
    """Write all outputs to structured Excel file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        kpi_df.to_excel(
            writer, sheet_name='KPI_Summary', index=False)
        city_df.to_excel(
            writer, sheet_name='City_Performance', index=False)
        income_df.to_excel(
            writer, sheet_name='Income_Segments', index=False)
        credit_df.to_excel(
            writer, sheet_name='Credit_Analysis', index=False)
        monthly_df.to_excel(
            writer, sheet_name='Monthly_Trend', index=False)
        age_df.to_excel(
            writer, sheet_name='Age_Group_Analysis', index=False)
        health_df.to_excel(
            writer, sheet_name='Financial_Health', index=False)
        savings_df.to_excel(
            writer, sheet_name='Savings_Segments', index=False)
        alerts_df.to_excel(
            writer, sheet_name='At_Risk_Alerts', index=False)
        churn_df.to_excel(
            writer, sheet_name='Churn_Prediction', index=False)
        clean_df.to_excel(
            writer, sheet_name='Transaction_Data', index=False)
 
    logging.info(f"Excel report written -> {output_path}")
    print(f"Excel saved -> {output_path}")
 
 
def run_pipeline():
    """Master function — runs everything in order."""
    print("\n" + "="*55)
    print("  FINTECH SAVINGS INTELLIGENCE PIPELINE")
    print(f"  {TODAY.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55 + "\n")
 
    df = load_data(INPUT_FILE)
    df = clean_and_validate(df)
    df = engineer_features(df)
    df, alerts_df = detect_anomalies(df)
    df, churn_df  = predict_churn(df)
    kpi_df = calculate_kpis(df)
    (city_df, income_df, credit_df, monthly_df,
     age_df, health_df, savings_df) = build_summary_tables(df)
    export_to_excel(
        kpi_df, city_df, income_df, credit_df,
        monthly_df, age_df, health_df, savings_df,
        alerts_df, churn_df, df, OUTPUT_FILE
    )
 
    print("\n" + "="*55)
    print("  PIPELINE COMPLETE")
    print(f"  Output -> {OUTPUT_FILE}")
    print(f"  Log    -> {LOG_FILE}")
    print("="*55 + "\n")
    logging.info("PIPELINE COMPLETED SUCCESSFULLY\n")

# ----------------------------------------------------------
# SECTION 9: ML CHURN PREDICTION
# ----------------------------------------------------------
def predict_churn(df):
    """
    Train a Random Forest model to predict which clients
    are likely to become financially stressed next month.
    """
    print("Training ML churn prediction model...")

    features = [
        'credit_score',
        'debt_to_income',
        'yearly_income',
        'savings_capacity',
        'num_credit_cards',
        'current_age',
        'financial_health_score'
    ]
    target = 'financial_stress_flag'

    df_ml = df[features + [target]].copy()
    df_ml = df_ml.dropna()

    X = df_ml[features]
    y = df_ml[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.1f}%")
    logging.info(f"ML Model accuracy: {accuracy * 100:.1f}%")

    X_all = df[features].fillna(0)
    df['churn_probability'] = (
        model.predict_proba(X_all)[:, 1] * 100
    ).round(1)

    model_path = "D:/Jar_Financial_Intelligence/scripts/churn_model.pkl"
    joblib.dump(model, model_path)

    churn_output = df.groupby('client_id').agg(
        City              = ('merchant_city', 'first'),
        Age               = ('current_age', 'first'),
        Yearly_Income     = ('yearly_income', 'first'),
        Credit_Score      = ('credit_score', 'first'),
        Health_Score      = ('financial_health_score', 'mean'),
        Churn_Probability = ('churn_probability', 'mean')
    ).reset_index().round(2)

    churn_output['Churn_Risk_Label'] = pd.cut(
        churn_output['Churn_Probability'],
        bins=[-1, 20, 50, 75, 101],
        labels=['Low Risk', 'Medium Risk',
                'High Risk', 'Critical Risk']
    )

    churn_output.sort_values(
        'Churn_Probability', ascending=False, inplace=True
    )

    critical = (churn_output['Churn_Risk_Label'] == 'Critical Risk').sum()
    high = (churn_output['Churn_Risk_Label'] == 'High Risk').sum()

    print(f"Critical Risk clients: {critical:,}")
    print(f"High Risk clients: {high:,}")
    print("ML churn prediction complete")

    return df, churn_output 
 
# -- Entry point --------------------------------------------
if __name__ == "__main__":
    run_pipeline()