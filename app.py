import streamlit as st
import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)


@st.cache_data
def load_and_clean_data(uploaded_file):
    """Loads and performs initial cleaning of the dataset."""
    df = pd.read_csv(uploaded_file)
    df.dropna(subset=['Customer ID'], inplace=True)
    df = df[~df['Invoice'].str.startswith('C', na=False)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    return df

@st.cache_data
def process_data_for_model(_df):
    """
    Processes the data to create RFM features, churn labels, and clusters for visualization.
    This is the core logic for a realistic churn prediction model.
    """
    cutoff_date = _df['InvoiceDate'].max() - dt.timedelta(days=90)
    df_train = _df[_df['InvoiceDate'] < cutoff_date]
    df_future = _df[_df['InvoiceDate'] >= cutoff_date]

    rfm = df_train.groupby('Customer ID').agg({
        'InvoiceDate': lambda date: (cutoff_date - date.max()).days, 
        'Invoice': 'nunique',                                    
        'TotalPrice': 'sum'                                     
    })
    rfm.rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'TotalPrice': 'MonetaryValue'}, inplace=True)

    future_customers = df_future['Customer ID'].unique()
    rfm['Churn'] = rfm.index.isin(future_customers).astype(int)
    rfm['Churn'] = rfm['Churn'].apply(lambda x: 0 if x == 1 else 1) 
  
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'MonetaryValue']])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm

def train_and_save_model(processed_df):
    """Trains the model using the processed RFM data and saves it to a file."""
    X = processed_df[['Recency', 'Frequency', 'MonetaryValue']]
    y = processed_df['Churn']
    
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, 'churn_model.pkl')
    return model

with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file with transactional data", type="csv")

    st.header("2. Predict Churn")
    st.write("Enter customer RFM values to predict their churn probability.")
    
    recency_input = st.slider("Recency (Days since last purchase)", 0, 365, 50)
    frequency_input = st.slider("Frequency (Total purchases)", 1, 100, 5)
    monetary_input = st.slider("Monetary Value (Total spent)", 0, 5000, 300)
    
    predict_button = st.button("Predict Churn")

st.title("📈 Customer Analytics Dashboard")
st.write("This dashboard analyzes customer data to create segments and predict churn risk.")

if uploaded_file is not None:
   
    df_clean = load_and_clean_data(uploaded_file)
    rfm_processed = process_data_for_model(df_clean)
    model = train_and_save_model(rfm_processed)

    st.header("Customer Segmentation")
    st.write("Customers are segmented into 4 clusters based on their RFM values.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=rfm_processed,
        x='Recency',
        y='MonetaryValue',
        hue='Cluster',
        palette='viridis',
        size='Frequency',
        sizes=(20, 200),
        ax=ax
    )
    plt.title('Customer Segments based on RFM')
    st.pyplot(fig)
    
    st.write("### Sample of Processed Data")
    st.dataframe(rfm_processed.head())

else:
    st.info("Awaiting for a CSV file to be uploaded. Please use the sidebar to begin.")

if predict_button:
    if not os.path.exists('churn_model.pkl'):
        st.sidebar.error("Please upload data to train the model first.")
    else:
        model = joblib.load('churn_model.pkl')
        input_data = pd.DataFrame({
            'Recency': [recency_input], 
            'Frequency': [frequency_input], 
            'MonetaryValue': [monetary_input]
        })
        
        prediction_proba = model.predict_proba(input_data)[0][1] 
        
        st.sidebar.metric(
            label="Churn Probability",
            value=f"{prediction_proba:.2%}"
        )
        if prediction_proba > 0.5:
            st.sidebar.warning("High risk of churn!")
        else:
            st.sidebar.success("Low risk of churn.")
