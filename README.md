# PAIMANA: AI-Driven Infrastructure Early Warning System 🚀

**PAIMANA** is an executive-level Early Warning System (EWS) designed to monitor, predict, and explain cost overruns and time delays in large-scale infrastructure projects. Powered by an ensemble of gradient-boosted trees and explainable AI (SHAP), it provides decision-makers with real-time risk intelligence.

## 🌟 Key Features

* **Executive Dashboard:** High-level KPIs, risk distributions, and regional bottlenecks at a glance.
* **Project Ledger:** A searchable, exportable database of all active infrastructure projects, complete with their AI-assigned Risk Tiers (Critical, High, Medium, Low).
* **Explainable AI Scenario Lab:** An interactive sandbox where users can adjust project parameters (budget, physical progress, timelines) and instantly see how the AI recalculates risk. Features live **Dual-SHAP Diagnostics** to mathematically explain exactly *why* a project is at risk of cost or time overruns.
* **AI Co-Pilot:** A heuristic "RAG-lite" neural chat interface that allows users to query the portfolio using natural language (e.g., *"Show me the most delayed projects in Bihar"* or *"Explain SHAP drivers for N04000050"*).

## 🛠️ Technology Stack

* **Backend:** Python, Flask, Gunicorn
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
* **Explainable AI:** SHAP (SHapley Additive exPlanations)
* **Data Processing:** Pandas, NumPy
* **Frontend:** HTML5, vanilla JavaScript, Tailwind CSS (Custom Glassmorphism UI)

## 📂 Repository Structure

\`\`\`text
├── templates/                 # HTML UI components (Base, Dashboard, Ledger, Scenario, Co-Pilot)
├── ewas_model.pkl             # Trained XGBoost/LightGBM ensemble and encoders
├── clean_features.zip         # Compressed raw feature matrix (zipped to bypass 100MB limits)
├── portfolio_summary.csv      # Lightweight cached dataset for fast UI rendering
├── requirements.txt           # Python dependencies for deployment
├── server.py                  # Main Flask application and API routing
└── README.md                  # Project documentation
\`\`\`

## 🚀 Live Deployment (Render)

This application is optimized for free deployment on [Render](https://render.com/). 
1. Connect this repository to a new Render **Web Service**.
2. Set the Build Command to: `pip install -r requirements.txt`
3. Set the Start Command to: `gunicorn server:app`
4. Deploy! *(Note: The Pandas engine handles the `.zip` extraction in memory at runtime).*

## 💻 Local Installation

To run this project on your local machine:

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/Bhojrajsahu07/SIH-2026.git 
   cd SIH-2026
   \`\`\`

2. **Install the required dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Boot the Flask server:**
   \`\`\`bash
   python server.py
   \`\`\`

4. **Access the application:**
   Open your web browser and navigate to `http://localhost:5000`.

---
*Developed for intelligent infrastructure monitoring and risk mitigation.*
