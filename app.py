import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline
import re
import ast
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# Page config for good impression (wide layout, green theme)
st.set_page_config(page_title="EVGen Pricer", layout="wide", initial_sidebar_state="expanded")

# Enhanced Custom CSS for Eye-Catching UI (Green Eco Theme) - Fixed Emoji Visibility
st.markdown("""
    <style>
    /* Global Theme */
    .main {background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%);}
    .stApp {background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%);}
    
    /* Header Styling */
    h1 {
        font-size: 3rem !important;
        background: linear-gradient(45deg, #4CAF50, #81C784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-weight: bold;
    }
    
    /* Eye-Catching Highlights Section */
    .highlights-card {
        background: linear-gradient(135deg, #4CAF50, #66BB6A);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.3);
        margin: 1rem 0;
        color: white;
    }
    .highlights-card h2 {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        color: white !important;  /* Solid white for emoji visibility */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);  /* Add shadow for pop */
        font-weight: bold;
    }
    .highlights-card ul {
        list-style: none;
        padding: 0;
    }
    .highlights-card li {
        font-size: 1.2rem;
        margin: 0.8rem 0;
        padding: 0.5rem;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .highlights-card li:hover {
        transform: translateX(10px);
        background: rgba(255,255,255,0.2);
    }
    .highlights-card li::before {
        content: '⚡';
        margin-right: 1rem;
        font-size: 1.5rem;
    }
    
    /* Enhanced Chat Styling - Friend-like Conversation Bubbles */
    .stChatMessage {
        margin-bottom: 1rem;
        border-radius: 18px;
        padding: 1rem;
        position: relative;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stChatMessage.user {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .stChatMessage.assistant {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9);
        margin-right: auto;
        border-bottom-left-radius: 4px;
        color: #2E7D32;
    }
    .stChatMessage:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    /* Avatar Styling */
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
        font-size: 1.5rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    .user-avatar { background: linear-gradient(45deg, #2196F3, #21CBF3); }
    .assistant-avatar { background: linear-gradient(45deg, #4CAF50, #8BC34A); }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #4CAF50, #81C784);
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: white !important;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.2) !important;
        transform: scale(1.05);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white !important;
        color: #4CAF50 !important;
        font-weight: bold;
    }
    
    /* Metrics Styling */
    .stMetric {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(76, 175, 80, 0.2);
        text-align: center;
    }
    .stMetric > div > div {
        color: #2E7D32 !important;
        font-size: 1.5rem !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #66BB6A);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    
    /* Sidebar Enhancement */
    .css-1d391kg {
        background: linear-gradient(180deg, #4CAF50, #2E7D32);
        padding: 1rem;
        border-radius: 0 15px 15px 0;
    }
    .css-1d391kg h1 {
        color: white !important;
        background: none !important;
        -webkit-text-fill-color: white !important;
    }
    
    /* Plotly Chart Enhancements */
    .plotly-chart {
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        overflow: hidden;
    }

    /* Footer Styling */
    .footer {
        background: linear-gradient(135deg, #2E7D32, #4CAF50);
        color: white;
        text-align: center;
        padding: 2rem;
        margin-top: 2rem;
        border-radius: 15px 15px 0 0;
        box-shadow: 0 -4px 16px rgba(0,0,0,0.1);
    }
    .footer h3 {
        color: white !important;
        margin-bottom: 1rem;
    }
    .footer a {
        color: #E8F5E8;
        text-decoration: none;
        font-weight: bold;
        margin: 0 1rem;
    }
    .footer a:hover {
        color: white;
        text-shadow: 0 0 5px rgba(255,255,255,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Load model & generator (cached)
@st.cache_resource
def load_model():
    model = joblib.load('models/ev_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    cols = joblib.load('models/cols.pkl')
    return model, preprocessor, cols

model, preprocessor, cols = load_model()
numerical_cols = cols['numerical_cols']
categorical_cols = cols['categorical_cols']

@st.cache_resource
def load_generator():
    return pipeline('text-generation', model='distilgpt2', device=-1)

generator = load_generator()

# Your functions (extract & explain - from before)
def extract_features(user_input):
    features = {
        'Brand': re.search(r'(tesla|bmw|vw|volkswagen|polestar|honda)', user_input, re.I).group(1).title() if re.search(r'(tesla|bmw|vw|volkswagen|polestar|honda)', user_input, re.I) else 'Generic',
        'Range_Km': int(re.search(r'(\d+(?:\.\d+)?)\s*(km|mile)', user_input, re.I).group(1)) if re.search(r'(\d+(?:\.\d+)?)\s*(km|mile)', user_input, re.I) else 300,
        'AccelSec': float(re.search(r'(\d+(?:\.\d+)?)\s*s(ec|accel|0-100)', user_input, re.I).group(1)) if re.search(r'(\d+(?:\.\d+)?)\s*s(ec|accel|0-100)', user_input, re.I) else 6.0,
        'TopSpeed_KmH': 200,
        'Efficiency_WhKm': 170,
        'Seats': 5,
        'FastCharge_KmH': 400,
        'PowerTrain': 'AWD' if 'awd' in user_input.lower() else 'RWD',
        'BodyStyle': re.search(r'(suv|sedan|hatchback|liftback)', user_input, re.I).group(1).title() if re.search(r'(suv|sedan|hatchback|liftback)', user_input, re.I) else 'Sedan',
        'Segment': 'D' if 'luxury' in user_input.lower() else 'C'
    }

    missing_keys = [k for k in features if features[k] in ['Generic', 300, 6.0, 170]]
    if missing_keys:
        prompt = f"""Refine EV features dict from query: '{user_input}'. Fill only missing: {missing_keys}.
        Output ONLY updated dict snippet, e.g., {{'Efficiency_WhKm': 165, 'TopSpeed_KmH': 220}}."""

        response = generator(prompt, max_length=100, temperature=0.3)[0]['generated_text']

        try:
            dict_match = re.search(r'\{.*\}', response, re.DOTALL)
            if dict_match:
                snippet = ast.literal_eval(dict_match.group(0))
                features.update(snippet)
        except:
            pass

    features['Efficiency_KmKWh'] = 1000 / features['Efficiency_WhKm']
    return features

def generate_explanation(price, features):
    prompt = f"Write a short EV price explanation (~50 words): Price €{price:.0f} for {features['Brand']} {features['BodyStyle']} with {features['Range_Km']}km range, {features['PowerTrain']}, {features['AccelSec']}s accel. Highlight cost drivers + green tip on efficiency {features['Efficiency_KmKWh']:.1f} km/kWh (higher = greener, lower emissions). End with eco suggestion."

    response = generator(prompt, max_length=120, num_return_sequences=1, temperature=0.7, do_sample=True)[0]['generated_text']

    explanation = response.replace(prompt, '').strip()
    explanation = ' '.join(explanation.split()[:25])

    if len(explanation) < 20:
        explanation = f"This {features['Brand']} EV costs ~€{price:.0f} due to range and features. Green tip: {features['Efficiency_KmKWh']:.1f} km/kWh efficiency saves CO2—pair with home solar!"

    return explanation

# Header with good impression
st.header("⚡ EVGen Pricer: Chat Your Way to Green Rides!")
st.write("Chat or tweak specs for price predictions + eco insights! Powered by Green AI (low-energy models).")

# Enhanced Highlights Section (Now a Gradient Card) - Emoji Fixed
st.markdown("""
<div class="highlights-card">
    <h2>🚀 Discover Your EV Worth!</h2>
    <ul>
        <li>Predict EV prices with a single click!</li>
        <li>Powered by Random Forest (R² up to 0.881) + DistilGPT2 GenAI!</li>
        <li>Visualize actual vs. predicted prices like a pro!</li>
        <li>Built with love using Python & Streamlit!</li>
        <li>Easy, fast, & fun for eco-buyers! (MAE: €6,822)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Tabs for UX: Chat | Interactive Specs | Charts
tab1, tab2, tab3 = st.tabs(["💬 Chat Mode", "⚙️ Tweak Specs", "📊 Visual Insights"])

with tab1:
    # Chat history - Enhanced with Avatars & Friend-like Bubbles
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Helper Visual: Friend Talking Intro (One-time Welcome)
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="😊"):
            st.markdown("""
            <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
                <div class="chat-avatar assistant-avatar">😊</div>
                <div>
                    <p>Hey there, eco-friend! 👋 I'm your EV buddy—tell me about your dream ride, like "a zippy Tesla SUV with 400km range," and I'll predict the price + green tips. What's on your mind? 🚀</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": "Hey there, eco-friend! 👋 I'm your EV buddy—tell me about your dream ride, like \"a zippy Tesla SUV with 400km range,\" and I'll predict the price + green tips. What's on your mind? 🚀"})

    for message in st.session_state.messages[1:]:  # Skip welcome if shown
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "😊"):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Describe your EV?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"""
            <div style="display: flex; align-items: flex-start;">
                <div class="chat-avatar user-avatar">👤</div>
                <div>{prompt}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="😊"):
            with st.spinner("🔄 Predicting..."):
                features = extract_features(prompt)
                test_df = pd.DataFrame([features])
                X_prep = test_df[numerical_cols + categorical_cols]
                X_processed = preprocessor.transform(X_prep)
                price = model.predict(X_processed)[0]
                explanation = generate_explanation(price, features)
                usd_price = price * 1.1
                response = f"**Predicted Cost: €{price:.0f} (~${usd_price:.0f})**\n\n{explanation}"
            st.markdown(f"""
            <div style="display: flex; align-items: flex-start;">
                <div class="chat-avatar assistant-avatar">😊</div>
                <div>{response}</div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    # Interactive sliders/dropdowns for features (UX fun! Updated with dataset options)
    st.subheader("Tweak Your EV Specs")
    col1, col2 = st.columns(2)
    with col1:
        range_km = st.slider("Range (km)", 100, 600, 300)
        accel_sec = st.slider("Accel (0-100s)", 3.0, 12.0, 6.0)
        seats = st.select_slider("Seats", options=[4, 5, 7])
    with col2:
        # Updated: Brands from dataset (Tesla, Volkswagen, Polestar, BMW, Honda + Generic)
        brand = st.selectbox("Brand", ['Tesla', 'Volkswagen', 'Polestar', 'BMW', 'Honda', 'Generic'])
        # Updated: Body Styles from dataset (SUV, Sedan, Hatchback, Liftback)
        body_style = st.selectbox("Body Style", ['SUV', 'Sedan', 'Hatchback', 'Liftback'])
        # Updated: Powertrain from dataset (AWD, RWD)
        power_train = st.selectbox("Powertrain", ['AWD', 'RWD'])

    if st.button("🔮 Predict Price", type="primary"):
        features = {
            'Brand': brand, 'Range_Km': range_km, 'AccelSec': accel_sec, 'TopSpeed_KmH': 200,
            'Efficiency_WhKm': 170, 'Seats': seats, 'FastCharge_KmH': 400, 'PowerTrain': power_train,
            'BodyStyle': body_style, 'Segment': 'D', 'Efficiency_KmKWh': 5.88
        }
        test_df = pd.DataFrame([features])
        X_prep = test_df[numerical_cols + categorical_cols]
        X_processed = preprocessor.transform(X_prep)
        price = model.predict(X_processed)[0]
        usd_price = price * 1.1
        eco_score = features['Efficiency_KmKWh']
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted Cost", f"€{price:.0f}", f"${usd_price:.0f}")
        with col_b:
            st.metric("Eco Score (km/kWh)", f"{eco_score:.1f}", "Higher = Greener!")
        st.write(generate_explanation(price, features))

# Updated Tab 3: Enhanced Visual Insights with More Intuitive Charts
with tab3:
    # Interactive Charts (load sample data for demo)
    st.subheader("Visual Insights")
    
    # Fixed load_sample_data with np import & error handling
    @st.cache_data
    def load_sample_data():
        try:
            df = pd.read_csv('data/ElectricCarData_Clean.csv')
            # Quick clean like before
            for col in ['AccelSec', 'TopSpeed_KmH', 'Range_Km', 'Efficiency_WhKm', 'FastCharge_KmH', 'Seats']:
                if col in df.columns:
                    df[col] = df[col].replace('-', np.nan).astype(float)
            df = df.dropna()
            df['Efficiency_KmKWh'] = 1000 / df['Efficiency_WhKm']
            return df
        except Exception as e:
            st.error(f"Chart data load error: {e}. Using dummy data.")
            # Enhanced dummy data to match full columns for better chart compatibility
            dummy_df = pd.DataFrame({
                'Brand': ['Tesla', 'BMW', 'Volkswagen', 'Polestar', 'Honda', 'Tesla', 'BMW', 'Volkswagen'],
                'Model': ['Model Y', 'i4', 'ID.3', '2', 'e:Ny1', 'Model 3', 'iX', 'ID.4'],
                'AccelSec': [5.0, 5.7, 7.9, 6.8, 8.5, 4.6, 4.6, 6.2],
                'TopSpeed_KmH': [217, 200, 160, 205, 160, 233, 250, 180],
                'Range_Km': [533, 590, 425, 479, 396, 491, 630, 520],
                'Efficiency_WhKm': [161, 189, 166, 181, 175, 161, 212, 185],
                'FastCharge_KmH': [595, 315, 170, 270, 100, 595, 195, 215],
                'PowerTrain': ['AWD', 'RWD', 'RWD', 'RWD', 'FWD', 'AWD', 'AWD', 'RWD'],
                'BodyStyle': ['SUV', 'Sedan', 'Hatchback', 'Liftback', 'SUV', 'Sedan', 'SUV', 'SUV'],
                'Segment': ['D', 'D', 'C', 'C', 'B', 'D', 'E', 'D'],
                'Seats': [5, 5, 5, 5, 5, 5, 5, 5],
                'PriceEuro': [65000, 75000, 38000, 43000, 35000, 50000, 85000, 45000],
                'RapidCharge': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
                'PlugType': ['IEC 62196-3', 'IEC 62196-3', 'Type 2', 'CCS', 'CHAdeMO', 'IEC 62196-3', 'CCS', 'CCS'],
                'Efficiency_KmKWh': [6.2, 5.3, 6.0, 5.5, 5.7, 6.2, 4.7, 5.4]
            })
            return dummy_df

    df_sample = load_sample_data()

    # Chart 1: Brand Distribution Bar - Simple Market Share Insight (Replaced Pie with Horizontal Bar for Better Comparison)
    if 'Brand' in df_sample.columns:
        brand_counts = df_sample['Brand'].value_counts().reset_index()
        brand_counts.columns = ['Brand', 'Count']
        brand_counts['Percentage'] = (brand_counts['Count'] / brand_counts['Count'].sum() * 100).round(1)
        
        fig_brand_bar = px.bar(brand_counts, x='Count', y='Brand', 
                               orientation='h',  # Horizontal for easy reading
                               title="EV Brand Market Shares (Dataset Snapshot)",
                               text='Percentage',  # Show % on bars
                               labels={'Count': 'Number of Models', 'Brand': 'Brand'})
        fig_brand_bar.update_traces(textposition='outside', texttemplate='%{text}%')
        fig_brand_bar.update_layout(height=400, xaxis_title="Number of Models", yaxis_title="Brand")
        st.plotly_chart(fig_brand_bar, use_container_width=True)

    # Chart 2: Average Price by Brand Bar - Quick Cost Comparison
    if 'Brand' in df_sample.columns and 'PriceEuro' in df_sample.columns:
        avg_price_by_brand = df_sample.groupby('Brand')['PriceEuro'].mean().reset_index()
        fig_price_bar = px.bar(avg_price_by_brand, x='Brand', y='PriceEuro', 
                               title="Average Price per Brand (€)", color='PriceEuro', 
                               color_continuous_scale='Greens')
        fig_price_bar.update_layout(height=400, xaxis_tickangle=45)
        st.plotly_chart(fig_price_bar, use_container_width=True)

    # Chart 3: Enhanced Interactive Scatter - Range vs Price (Size=Efficiency, Color=BodyStyle)
    if all(col in df_sample.columns for col in ['Range_Km', 'PriceEuro', 'Efficiency_KmKWh', 'BodyStyle']):
        fig_scatter = px.scatter(df_sample, x='Range_Km', y='PriceEuro', 
                                 color='BodyStyle', size='Efficiency_KmKWh',
                                 hover_data=['Brand', 'Model', 'Segment'],
                                 title="Range vs Price: Bigger Bubbles = Better Efficiency (Greener Rides!)",
                                 labels={'PriceEuro': 'Price (€)', 'Range_Km': 'Range (km)'})
        fig_scatter.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Chart 4: Boxplot Price by Segment - Segment Pricing Spread
    if all(col in df_sample.columns for col in ['PriceEuro', 'Segment']):
        fig_box = px.box(df_sample, x='Segment', y='PriceEuro', 
                         color='Segment',
                         title="Price Distribution by Car Segment (A=Small, E=Luxury)",
                         labels={'PriceEuro': 'Price (€)'})
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)

    # Chart 5: Correlation Heatmap - Numerical Feature Relationships
    numerical_features = ['AccelSec', 'TopSpeed_KmH', 'Range_Km', 'Efficiency_WhKm', 'FastCharge_KmH', 'Seats', 'PriceEuro']
    num_cols_available = [col for col in numerical_features if col in df_sample.columns]
    if len(num_cols_available) > 1:
        corr_matrix = df_sample[num_cols_available].corr()
        fig_heatmap = px.imshow(corr_matrix, aspect="auto", color_continuous_scale='RdBu_r',
                                title="Correlation Heatmap: How Features Relate to Price (Red=Positive, Blue=Negative)")
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Chart 6: Feature Importance Bar (from your model) - What Drives Prices?
    try:
        feature_names = numerical_cols + [f'cat__{name.replace(" ", "_")}' for name in preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)]
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        fig_importance = px.bar(importances, orientation='h', title="Key Price Drivers: Feature Importances (Higher = Bigger Impact)",
                                color=importances.values, color_continuous_scale='Greens')
        fig_importance.update_layout(height=500)
        st.plotly_chart(fig_importance, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance chart setup: {e}. Model ready, but skipping advanced viz.")

    # Insight Summary Text - Simple Takeaways
    st.markdown("""
    ### 🔍 Quick Insights from the Data
    - **Market Leaders**: Tesla often dominates shares & premium pricing—great for range but watch efficiency!
    - **Value Picks**: C/D segments offer balanced range (300-500km) under €50K with 5-6 km/kWh eco-scores.
    - **Trend Alert**: Longer range correlates with higher prices (r~0.6), but efficient models (low Wh/km) buck the trend for green savings.
    - **Pro Tip**: Filter by BodyStyle in scatter—SUVs pack power but sedans sip energy best!
    """)

# Footer Section - Moved Outside Sidebar
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>👨‍💼 About the Developer</h3>
    <p><strong>Dhruv Kumar Singh</strong></p>
    <p><a href="https://github.com/dhruvDS13" target="_blank">🌐 GitHub</a> | <a href="https://www.linkedin.com/in/dhruv-kumar-singh-51a86725a" target="_blank">💼 LinkedIn</a></p>
    <p style="font-size: 0.9rem; opacity: 0.9;">Built with passion for sustainable tech! Feel free to connect or collaborate. 🚀</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🌿 Green AI")
    
    # Live EV News Section - Fetches top headlines from InsideEVs RSS (updates every 10 min)
    @st.cache_data(ttl=600)  # Cache for 10 minutes to keep it "live" without overloading
    def fetch_ev_news():
        try:
            rss_url = "https://insideevs.com/feed/"  # Fixed: Actual RSS feed URL (was /rss/ directory)
            response = requests.get(rss_url, timeout=10)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}  # RSS is often Atom for this feed
            
            items = []
            for item in root.findall('.//atom:entry', ns) or root.findall('.//item'):  # Fallback for RSS1
                title_elem = item.find('atom:title', ns) or item.find('title')
                link_elem = item.find('atom:link', ns) or item.find('link')
                desc_elem = item.find('atom:summary', ns) or item.find('description')
                
                if title_elem is not None and link_elem is not None:
                    title = title_elem.text.strip() if title_elem.text else "No Title"
                    link = link_elem.get('href', link_elem.text) if hasattr(link_elem, 'get') else link_elem.text
                    desc = (desc_elem.text.strip()[:150] + "..." if desc_elem is not None and desc_elem.text else "Read more for details.")
                    
                    items.append({'title': title, 'link': link, 'description': desc})
                    if len(items) >= 5:  # Limit to top 5
                        break
            
            return items
        except Exception as e:
            st.error(f"News fetch error: {e}. Showing placeholder.")
            return [
                {'title': 'Tesla Cybertruck Update', 'link': 'https://insideevs.com', 'description': 'Latest on affordable EV trucks...'},
                {'title': 'EV Battery Breakthrough', 'link': 'https://insideevs.com', 'description': 'New solid-state tech promises 1000km range...'}
            ]

    # Display News
    st.markdown("---")
    with st.expander("📰 Latest EV News", expanded=False):
        news_items = fetch_ev_news()
        for i, news in enumerate(news_items, 1):
            st.markdown(f"**{i}. [{news['title']}]({news['link']})**")
            st.caption(news['description'])
            st.markdown("---")  # Thin separator
        
        st.caption(f"*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Source: InsideEVs*")
    st.markdown("---")
    with st.expander("👨‍💼 About the Developer", expanded=False):
            st.markdown("**Dhruv Kumar Singh**")
            st.markdown("[🌐 GitHub](https://github.com/dhruvDS13)")
            st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/dhruv-kumar-singh-51a86725a)")
            st.caption("Built with passion for sustainable tech! Feel free to connect or collaborate. 🚀")