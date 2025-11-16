import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from itertools import product
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ad Click Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .metric-card { padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; }
    .metric-value { font-size: 28px; font-weight: bold; }
    .metric-label { font-size: 14px; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

st.title("Ad Click Prediction - Advanced Dashboard")
st.markdown("---")

@st.cache_data
def load_data():
    data = pd.read_csv('sample_dataset.csv')
    np.random.seed(42)
    data['conversion'] = ((data['click'] == 1) & (np.random.random(len(data)) < 0.08)).astype(int)
    data = data.drop(columns=['id','full_name'], axis=1)
    data['gender'] = data['gender'].fillna('Unknown')
    data['device_type'] = data['device_type'].fillna('Unknown')
    data['ad_position'] = data['ad_position'].fillna('Unknown')
    data['browsing_history'] = data['browsing_history'].fillna('Unknown')
    data['time_of_day'] = data['time_of_day'].fillna('Unknown')
    data['age'] = data['age'].fillna(data['age'].median()).astype(int)
    data['age_group'] = pd.cut(data['age'], bins=[0, 24, 34, 44, 54, 64, 100], 
                               labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
    return data

data = load_data()

st.sidebar.header("Filters")
st.sidebar.markdown("---")

selected_devices = st.sidebar.multiselect("Device Type", data['device_type'].unique(), default=data['device_type'].unique())
selected_age = st.sidebar.multiselect("Age Group", data['age_group'].cat.categories.tolist(), default=data['age_group'].cat.categories.tolist())
selected_history = st.sidebar.multiselect("Browsing History", data['browsing_history'].unique(), default=data['browsing_history'].unique())
selected_time = st.sidebar.multiselect("Time of Day", data['time_of_day'].unique(), default=data['time_of_day'].unique())
selected_position = st.sidebar.multiselect("Ad Position", data['ad_position'].unique(), default=data['ad_position'].unique())
selected_gender = st.sidebar.multiselect("Gender", data['gender'].unique(), default=data['gender'].unique())

st.sidebar.markdown("---")
conversion_value = st.sidebar.slider("Conversion Value ($)", 10, 500, 200, 10)
click_value = st.sidebar.slider("Click Value ($)", 0, 20, 5, 1)
total_spend = st.sidebar.slider("Total Spend ($)", 5000, 100000, 20000, 1000)

filtered_data = data[
    (data['device_type'].isin(selected_devices)) &
    (data['age_group'].isin(selected_age)) &
    (data['browsing_history'].isin(selected_history)) &
    (data['time_of_day'].isin(selected_time)) &
    (data['ad_position'].isin(selected_position)) &
    (data['gender'].isin(selected_gender))
]

st.sidebar.info(f"Filtered: {len(filtered_data)} / {len(data)} records")

# Calculate main metrics
total_impressions = len(filtered_data)
total_clicks = filtered_data['click'].sum()
total_conversions = filtered_data['conversion'].sum()
revenue_conversions = total_conversions * conversion_value
revenue_clicks = (total_clicks - total_conversions) * click_value
total_revenue = revenue_conversions + revenue_clicks
ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
cvr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
roas = (total_revenue / total_spend) if total_spend > 0 else 0
cpa = (total_spend / total_conversions) if total_conversions > 0 else 0
ltv = (total_revenue / total_conversions) if total_conversions > 0 else 0
profit = total_revenue - total_spend
roi_pct = (profit / total_spend * 100) if total_spend > 0 else 0

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Metrics", "Analysis", "Funnels", "Segments", "Top/Bottom Performers", "EDA", "Data"])

# TAB 1: Metrics
with tab1:
    st.header("Key Metrics Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Impressions", f"{total_impressions:,}")
    with col2:
        st.metric("Clicks", f"{total_clicks:,}", delta=f"{ctr:.2f}%")
    with col3:
        st.metric("Conversions", f"{total_conversions:,}", delta=f"{cvr:.2f}%")
    with col4:
        st.metric("Revenue", f"${total_revenue:,.0f}", delta=f"ROAS: {roas:.2f}x")
    with col5:
        st.metric("ROI", f"{roi_pct:.1f}%", delta=f"Profit: ${profit:,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("AdTech Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['CTR', 'CVR', 'ROAS', 'CPA', 'LTV', 'Profit'],
            'Value': [f"{ctr:.2f}%", f"{cvr:.2f}%", f"{roas:.3f}", f"${cpa:.2f}", f"${ltv:.2f}", f"${profit:,.0f}"]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Conversion Funnel")
        fig = go.Figure(go.Funnel(
            y=['Impressions', 'Clicks', 'Conversions'],
            x=[total_impressions, total_clicks, total_conversions],
            textposition='inside',
            textinfo='value+percent initial',
            marker=dict(color=['#667eea', '#764ba2', '#f093fb'])
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: Analysis
with tab2:
    st.header("Segment Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        device_data = filtered_data.groupby('device_type').agg({'click': ['sum', 'count']}).reset_index()
        device_data.columns = ['Device', 'Clicks', 'Impressions']
        device_data['CTR'] = (device_data['Clicks'] / device_data['Impressions'] * 100).round(2)
        fig = px.bar(device_data, x='Device', y='CTR', color='CTR', title="CTR by Device Type")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        position_data = filtered_data.groupby('ad_position').agg({'click': ['sum', 'count']}).reset_index()
        position_data.columns = ['Position', 'Clicks', 'Impressions']
        position_data['CTR'] = (position_data['Clicks'] / position_data['Impressions'] * 100).round(2)
        fig = px.bar(position_data, x='Position', y='CTR', color='CTR', title="CTR by Ad Position")
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_data = filtered_data.groupby('time_of_day').agg({'click': ['sum', 'count']}).reset_index()
        time_data.columns = ['Time', 'Clicks', 'Impressions']
        time_data['CTR'] = (time_data['Clicks'] / time_data['Impressions'] * 100).round(2)
        fig = px.bar(time_data, x='Time', y='CTR', color='CTR', title="CTR by Time of Day")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        gender_data = filtered_data.groupby('gender').agg({'click': ['sum', 'count']}).reset_index()
        gender_data.columns = ['Gender', 'Clicks', 'Impressions']
        gender_data['CTR'] = (gender_data['Clicks'] / gender_data['Impressions'] * 100).round(2)
        fig = px.bar(gender_data, x='Gender', y='CTR', color='CTR', title="CTR by Gender")
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: Funnels
with tab3:
    st.header("Funnel Analysis")
    funnel_feature = st.selectbox("Select Feature:", ['device_type', 'gender', 'time_of_day', 'ad_position', 'browsing_history', 'age_group'])
    
    categories = filtered_data[funnel_feature].unique()
    num_cats = len(categories)
    cols = min(3, num_cats)
    rows = (num_cats + cols - 1) // cols
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=categories, specs=[[{'type': 'funnel'} for _ in range(cols)] for _ in range(rows)])
    
    colors = ['#667eea', '#764ba2', '#f093fb']
    labels = ['Impressions', 'Clicks', 'Conversions']
    
    for idx, cat in enumerate(categories):
        cat_data = filtered_data[filtered_data[funnel_feature] == cat]
        row = idx // cols + 1
        col = idx % cols + 1
        
        fig.add_trace(
            go.Funnel(
                y=labels,
                x=[len(cat_data), cat_data['click'].sum(), cat_data['conversion'].sum()],
                marker=dict(color=colors),
                textposition='inside',
                textinfo='value',
                textfont=dict(size=11, color='white'),
                name=str(cat),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
    
    for i in range(num_cats):
        if i > 0:
            fig.update_yaxes(visible=False, row=(i // cols + 1), col=(i % cols + 1))
    
    height = 600 if rows == 1 else (900 if rows == 2 else 1200)
    fig.update_layout(height=height, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: Segment Performance
with tab4:
    st.header("Segment Performance Metrics")
    
    segment_data = []
    categorical_features = ['device_type', 'gender', 'ad_position', 'browsing_history', 'time_of_day']
    
    for feature in categorical_features:
        for category in filtered_data[feature].unique():
            cat_data = filtered_data[filtered_data[feature] == category]
            impressions = len(cat_data)
            clicks = cat_data['click'].sum()
            conversions = cat_data['conversion'].sum()
            
            rev_conv = conversions * conversion_value
            rev_click = (clicks - conversions) * click_value
            total_rev = rev_conv + rev_click
            segment_spend = (impressions / len(filtered_data)) * total_spend if len(filtered_data) > 0 else 0
            
            ctr_seg = (clicks / impressions * 100) if impressions > 0 else 0
            cvr_seg = (conversions / clicks * 100) if clicks > 0 else 0
            roas_seg = (total_rev / segment_spend) if segment_spend > 0 else 0
            cpa_seg = (segment_spend / conversions) if conversions > 0 else 0
            ltv_seg = (total_rev / conversions) if conversions > 0 else 0
            profit_seg = total_rev - segment_spend
            roi_seg = (profit_seg / segment_spend * 100) if segment_spend > 0 else 0
            
            segment_data.append({
                'Feature': feature,
                'Segment': category,
                'Impressions': impressions,
                'Clicks': clicks,
                'Conversions': conversions,
                'CTR': round(ctr_seg, 2),
                'CVR': round(cvr_seg, 2),
                'ROAS': round(roas_seg, 3),
                'CPA': round(cpa_seg, 2),
                'LTV': round(ltv_seg, 2),
                'Spend': round(segment_spend, 2),
                'Profit': round(profit_seg, 2),
                'ROI': round(roi_seg, 2)
            })
    
    segment_df = pd.DataFrame(segment_data).sort_values('ROAS', ascending=False)
    st.dataframe(segment_df, use_container_width=True, height=400)

# TAB 5: Top/Bottom Performers
with tab5:
    st.header("Top and Bottom Segments")
    
    # Recalculate segments for this tab
    segment_data = []
    categorical_features = ['device_type', 'gender', 'ad_position', 'browsing_history', 'time_of_day']
    
    for feature in categorical_features:
        for category in filtered_data[feature].unique():
            cat_data = filtered_data[filtered_data[feature] == category]
            impressions = len(cat_data)
            clicks = cat_data['click'].sum()
            conversions = cat_data['conversion'].sum()
            
            rev_conv = conversions * conversion_value
            rev_click = (clicks - conversions) * click_value
            total_rev = rev_conv + rev_click
            segment_spend = (impressions / len(filtered_data)) * total_spend if len(filtered_data) > 0 else 0
            
            ctr_seg = (clicks / impressions * 100) if impressions > 0 else 0
            cvr_seg = (conversions / clicks * 100) if clicks > 0 else 0
            ltv_seg = (total_rev / conversions) if conversions > 0 else 0
            
            segment_data.append({
                'Segment': f"{feature}: {category}",
                'CTR': round(ctr_seg, 2),
                'LTV': round(ltv_seg, 2),
                'Conversions': conversions,
                'Spend': round(segment_spend, 2)
            })
    
    segment_df = pd.DataFrame(segment_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Segments by CTR")
        top_ctr = segment_df.nlargest(10, 'CTR')
        fig = px.bar(top_ctr, x='CTR', y='Segment', orientation='h', title="Top 10 Segments by CTR", color='CTR')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Bottom 10 Segments by CTR")
        bottom_ctr = segment_df.nsmallest(10, 'CTR')
        fig = px.bar(bottom_ctr, x='CTR', y='Segment', orientation='h', title="Bottom 10 Segments by CTR", color='CTR')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Segments by LTV")
        top_ltv = segment_df.nlargest(10, 'LTV')
        fig = px.bar(top_ltv, x='LTV', y='Segment', orientation='h', title="Top 10 Segments by LTV", color='LTV')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Bottom 10 Segments by LTV")
        bottom_ltv = segment_df.nsmallest(10, 'LTV')
        fig = px.bar(bottom_ltv, x='LTV', y='Segment', orientation='h', title="Bottom 10 Segments by LTV", color='LTV')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Segments by Conversions")
        top_conv = segment_df.nlargest(10, 'Conversions')
        fig = px.bar(top_conv, x='Conversions', y='Segment', orientation='h', title="Top 10 Segments by Conversions", color='Conversions')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Segments by Spend")
        top_spend = segment_df.nlargest(10, 'Spend')
        fig = px.bar(top_spend, x='Spend', y='Segment', orientation='h', title="Top 10 Segments by Spend", color='Spend')
        st.plotly_chart(fig, use_container_width=True)

# TAB 6: EDA
with tab6:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Click Distribution by Gender")
        gender_clicks = filtered_data.groupby('gender')['click'].sum().reset_index()
        fig = px.pie(gender_clicks, values='click', names='gender', title="Click Distribution by Gender")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Click Distribution by Device")
        device_clicks = filtered_data.groupby('device_type')['click'].sum().reset_index()
        fig = px.pie(device_clicks, values='click', names='device_type', title="Click Distribution by Device")
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        fig = px.histogram(filtered_data, x='age', nbins=20, title="Age Distribution", color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Conversions by Age Group")
        age_conv = filtered_data.groupby('age_group')['conversion'].sum().reset_index()
        fig = px.bar(age_conv, x='age_group', y='conversion', title="Conversions by Age Group", color='conversion')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Browsing History Distribution")
        history_dist = filtered_data['browsing_history'].value_counts().reset_index()
        history_dist.columns = ['History', 'Count']
        fig = px.bar(history_dist, x='History', y='Count', title="Records by Browsing History")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Ad Position Performance")
        position_data = filtered_data.groupby('ad_position').agg({'click': 'sum', 'conversion': 'sum'}).reset_index()
        fig = px.scatter(position_data, x='click', y='conversion', size='click', text='ad_position', title="Position: Clicks vs Conversions")
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Time of Day - Clicks and Conversions")
        time_data = filtered_data.groupby('time_of_day').agg({'click': 'sum', 'conversion': 'sum'}).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=time_data['time_of_day'], y=time_data['click'], name='Clicks'), secondary_y=False)
        fig.add_trace(go.Scatter(x=time_data['time_of_day'], y=time_data['conversion'], name='Conversions', mode='lines+markers'), secondary_y=True)
        fig.update_layout(title="Time of Day Performance", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Device Type - Clicks and Conversions")
        device_data = filtered_data.groupby('device_type').agg({'click': 'sum', 'conversion': 'sum'}).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=device_data['device_type'], y=device_data['click'], name='Clicks'), secondary_y=False)
        fig.add_trace(go.Scatter(x=device_data['device_type'], y=device_data['conversion'], name='Conversions', mode='lines+markers'), secondary_y=True)
        fig.update_layout(title="Device Type Performance", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

# TAB 7: Data View
with tab7:
    st.header("Raw Data")
    columns_to_show = st.multiselect(
        "Select columns:",
        options=filtered_data.columns.tolist(),
        default=['device_type', 'gender', 'age_group', 'time_of_day', 'click', 'conversion']
    )
    
    st.dataframe(filtered_data[columns_to_show].head(100), use_container_width=True, height=400)
    
    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_data.to_csv(index=False),
        file_name="filtered_data.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'><p>Ad Click Prediction Dashboard | Built with Streamlit and Plotly</p></div>", unsafe_allow_html=True)