# EVGen-Pricer

[![Streamlit](https://img.shields.io/badge/Streamlit-FF6B35?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/) [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

## 📝 Overview
EVGen-Pricer is a prototype web application designed to explore electric vehicle (EV) price estimation through user-friendly interfaces. It combines basic machine learning techniques with interactive elements to help users understand factors influencing EV costs. This is an experimental project developed during an ongoing internship, focusing on sustainable tech and data-driven insights.

**Note**: This repository contains early-stage code and models. It's a work-in-progress—feedback welcome, but not for production use yet.

## 🏗️ Problem Statement
Prospective EV buyers face confusion from hundreds of models, juggling range, price, speed, and eco-factors without easy guidance. Static tools overwhelm with lists, causing indecision and slower adoption. Solution: A GenAI chatbot for natural chats, smart price predictions, green tips, and interactive visuals to simplify choices and promote sustainability.

## ✨ Key Features (Prototype)
- **Interactive Spec Adjustment**: Tweak EV attributes (e.g., range, brand) via sliders and dropdowns.
- **Basic Visualizations**: Simple charts for exploring data trends like price vs. range.
- **Eco Insights**: Preliminary tips on energy efficiency (km/kWh) to encourage sustainable choices.
- **Live News Feed**: Sidebar with EV headlines for staying updated.

## 🛠 Tech Stack
- **Frontend**: Streamlit for rapid prototyping.
- **Backend/ML**: Scikit-learn for basic regression models; Hugging Face for lightweight text generation.
- **Data Handling**: Pandas, NumPy; Plotly for charts.
- **Environment**: Python 3.8+.

## 📁 Project Structure
```
EVGen-Pricer/
├── app.py                 # Main Streamlit script
├── models/                # Serialized model files (e.g., ev_model.pkl)
├── data/                  # Sample EV dataset (CSV)
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## 🚀 Getting Started
1. **Clone the Repo**:
   ```
   git clone https://github.com/dhruvDS13/EVGen-Pricer.git
   cd EVGen-Pricer
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run Locally**:
   ```
   streamlit run app.py
   ```
   Open [http://localhost:8501](http://localhost:8501) in your browser.

**Requirements**: Ensure you have the `models/` and `data/` folders populated (included in repo).

## 📄 License
MIT License © 2025. See [LICENSE](LICENSE) for details.

## 👨‍💻 Author
**Dhruv Kumar Singh**  
[GitHub](https://github.com/dhruvDS13) | [LinkedIn](https://www.linkedin.com/in/dhruv-kumar-singh-51a86725a)  

*Exploring green tech one prediction at a time. 🚀*
