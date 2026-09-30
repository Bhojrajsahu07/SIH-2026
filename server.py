import os, pickle, joblib, re
import pandas as pd
import numpy as np
import shap
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# -------------------------------------------------------------
# 1. Load Models & Datasets
# -------------------------------------------------------------
try:
    with open('ewas_model.pkl', 'rb') as f:
        ewas_dict = pickle.load(f)
except Exception:
    ewas_dict = joblib.load('ewas_model.pkl')

models = ewas_dict.get('models', {})
encoders = ewas_dict.get('encoders', {})
feature_columns = ewas_dict.get('feature_columns', [])

portfolio_df = pd.read_csv('portfolio_summary.csv')
clean_features_df = pd.read_csv('clean_features.csv') if os.path.exists('clean_features.csv') else None

# Dual SHAP Explainer Initialization for both targets
explainer_cost = shap.TreeExplainer(models['Cost_Overrun_Pct']['model_b'])
explainer_time = shap.TreeExplainer(models['Time_Overrun_Months']['model_b'])

# -------------------------------------------------------------
# 2. Intelligence Routines (Co-Pilot)
# -------------------------------------------------------------
def get_project_shap_explanation(project_id):
    if clean_features_df is None:
        return "<p class='text-slate-400 italic'>SHAP explainability tree is currently offline.</p>"

    try:
        matched = clean_features_df[clean_features_df['Project_ID'].astype(str).str.lower() == str(project_id).lower()]
        if matched.empty:
            return f"<p class='text-amber-400'>No raw feature matrix record found for Project ID: <b>{project_id}</b>.</p>"

        df_proc = matched.reindex(columns=feature_columns, fill_value=0).copy()
        for col, enc in encoders.items():
            if col in df_proc.columns:
                mapping = {cls: idx for idx, cls in enumerate(enc.classes_)}
                df_proc[col] = df_proc[col].map(mapping).fillna(-1).astype(int)

        X_vals = df_proc.astype(float).values
        
        shap_cost_vals = explainer_cost.shap_values(X_vals)[0]
        shap_time_vals = explainer_time.shap_values(X_vals)[0]

        top_cost = sorted([{"feat": col, "val": round(float(v), 2)} for col, v in zip(feature_columns, shap_cost_vals)], key=lambda x: abs(x['val']), reverse=True)[:3]
        top_time = sorted([{"feat": col, "val": round(float(v), 2)} for col, v in zip(feature_columns, shap_time_vals)], key=lambda x: abs(x['val']), reverse=True)[:3]

        def _render_bars(drivers, unit, color_class):
            html = "<div class='space-y-2'>"
            for d in drivers:
                bar_pct = min(100, max(15, int(abs(d['val']) * 2)))
                sign = '+' if d['val'] > 0 else ''
                text_col = 'text-red-400' if d['val'] > 0 else 'text-emerald-400'
                html += f"""
                <div>
                    <div class="flex justify-between text-xs text-slate-300 mb-1">
                        <span>{d['feat'].replace('_', ' ')}</span>
                        <span class="font-mono {text_col}">{sign}{d['val']} {unit}</span>
                    </div>
                    <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                        <div class="bg-gradient-to-r from-{color_class}-500 to-{color_class}-600 h-full rounded-full" style="width: {bar_pct}%"></div>
                    </div>
                </div>
                """
            return html + "</div>"

        return f"""
        <div class="mb-4 bg-slate-900/80 p-4 rounded-xl border border-blue-500/30">
            <div class="text-xs font-semibold text-blue-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                <i class="ph ph-chart-bar-horizontal"></i> SHAP Diagnostics for {project_id}
            </div>
            <div class="grid grid-cols-2 gap-6">
                <div>
                    <h4 class="text-[10px] text-slate-500 uppercase mb-2 border-b border-slate-700 pb-1">Cost Overrun Drivers</h4>
                    {_render_bars(top_cost, '%', 'blue')}
                </div>
                <div>
                    <h4 class="text-[10px] text-slate-500 uppercase mb-2 border-b border-slate-700 pb-1">Time Delay Drivers</h4>
                    {_render_bars(top_time, 'Mos', 'indigo')}
                </div>
            </div>
        </div>
        """
    except Exception as err:
        return f"<p class='text-rose-400'>Error calculating SHAP drivers: {str(err)}</p>"


def search_rag_context(query, top_k=5):
    query_str = str(query).lower()
    df_copy = portfolio_df.copy()
    df_copy['Predicted_Delay_Months'] = pd.to_numeric(df_copy['Predicted_Delay_Months'], errors='coerce').fillna(0)
    df_copy['Original_Cost'] = pd.to_numeric(df_copy['Original_Cost'], errors='coerce').fillna(0)
    df_copy['Risk_Score'] = df_copy['Risk_Tier'].map({'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}).fillna(0)

    # Synonyms and Intent Matching
    if any(k in query_str for k in ['best', 'least risk', 'lowest delay', 'efficient', 'safest', 'top']):
        matched = df_copy.sort_values(by=['Risk_Score', 'Predicted_Delay_Months'], ascending=[True, True])
        title = "Best Performing Projects"
    elif any(k in query_str for k in ['worst', 'highest risk', 'most delayed', 'critical', 'lagging', 'late', 'slow', 'bottleneck']):
        matched = df_copy.sort_values(by=['Risk_Score', 'Predicted_Delay_Months'], ascending=[False, False])
        title = "Highest Risk / Most Delayed Projects"
    elif any(k in query_str for k in ['highest budget', 'most expensive', 'largest cost', 'costly', 'funds', 'money']):
        matched = df_copy.sort_values(by=['Original_Cost'], ascending=False)
        title = "Highest Budget Projects"
    elif any(k in query_str for k in ['lowest budget', 'cheapest', 'smallest cost']):
        matched = df_copy.sort_values(by=['Original_Cost'], ascending=True)
        title = "Lowest Budget Projects"
    else:
        # Strict AND filtering for keyword tokens
        stop_words = {'project', 'projects', 'show', 'list', 'all', 'me', 'the', 'in', 'for', 'with', 'of', 'and', 'find', 'get', 'are', 'what', 'which'}
        tokens = [word for word in re.findall(r'\b\w+\b', query_str) if word not in stop_words]
        
        mask = pd.Series([True] * len(portfolio_df))
        for token in tokens:
            token_mask = (
                portfolio_df['State_Location'].astype(str).str.lower().str.contains(token, na=False) | 
                portfolio_df['Sector'].astype(str).str.lower().str.contains(token, na=False) | 
                portfolio_df['Risk_Tier'].astype(str).str.lower().str.contains(token, na=False) |
                portfolio_df['Project_ID'].astype(str).str.lower().str.contains(token, na=False)
            )
            mask = mask & token_mask # Using AND logic so filters stack
        
        matched = portfolio_df[mask]
        title = "Matched Search Results"

    if matched.empty:
        return [], ""

    records = matched.head(top_k).to_dict(orient='records')
    
    # Generate an LLM-style natural language summary
    avg_delay = round(sum(r.get('Predicted_Delay_Months', 0) for r in records) / len(records), 1)
    total_cost = round(sum(r.get('Original_Cost', 0) for r in records), 2)
    summary_text = f"I found {len(matched)} matching projects. Among the top {len(records)} shown below, the average delay is {avg_delay} months, with a combined budget of ₹{total_cost:,} Cr."

    return records, f"{title}|{summary_text}"

# -------------------------------------------------------------
# 3. Web Page Routes
# -------------------------------------------------------------
@app.route('/')
def dashboard():
    df = pd.read_csv('portfolio_summary.csv')
    kpis = {"total": f"{len(df):,}", "critical": f"{len(df[df['Risk_Tier'] == 'Critical']):,}", "cost": round(pd.to_numeric(df['Original_Cost'], errors='coerce').sum() / 100000, 2)}
    sec_agg = df.groupby('Sector')['Predicted_Delay_Months'].mean().reset_index().sort_values('Predicted_Delay_Months').tail(6)
    state_agg = df[~df['State_Location'].str.upper().isin(['UNKNOWN', 'NAN', 'N/A', '', '0'])].groupby('State_Location').size().reset_index(name='count').sort_values('count').tail(6)
    return render_template('dashboard.html', active_page='dashboard', kpis=kpis, sectors=sec_agg['Sector'].tolist(), delays=sec_agg['Predicted_Delay_Months'].tolist(), states=state_agg['State_Location'].tolist(), counts=state_agg['count'].tolist())

@app.route('/ledger')
def ledger():
    df = pd.read_csv('portfolio_summary.csv').replace([np.inf, -np.inf], np.nan).fillna("")
    return render_template('ledger.html', active_page='ledger', projects=df.head(100).astype(str).to_dict(orient='records'))

@app.route('/scenario')
def scenario():
    df = pd.read_csv('clean_features.csv')
    sectors = sorted([str(x) for x in df['Sector'].unique() if str(x).upper() not in ['UNKNOWN', 'NAN']])
    agencies = sorted([str(x) for x in df['Implementing_Agency'].unique() if str(x).upper() not in ['UNKNOWN', 'NAN']])
    return render_template('scenario.html', active_page='scenario', sectors=sectors, agencies=agencies)

@app.route('/copilot')
def copilot():
    return render_template('copilot.html', active_page='copilot')

# -------------------------------------------------------------
# 4. API Endpoints
# -------------------------------------------------------------
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        raw = request.json
        orig_cost = max(1.0, float(raw.get('Original_Cost', 1000.0)))
        cum_exp = float(raw.get('Cumulative_Expenditure', 0.0))
        phys_prog = float(raw.get('Physical_Progress_Pct', 0.0))
        
        try:
            app_d, tgt_d, rpt_d = pd.to_datetime(raw.get('Approval_Date')), pd.to_datetime(raw.get('Target_Date')), pd.to_datetime(raw.get('Report_Date'))
            orig_dur = max(1.0, float((tgt_d.year - app_d.year)*12 + tgt_d.month - app_d.month))
            elapsed = max(1.0, float((rpt_d.year - app_d.year)*12 + rpt_d.month - app_d.month))
        except:
            orig_dur, elapsed = 48.0, 24.0
            
        features = {
            'Sector': raw.get('Sector', 'ROAD TRANSPORT'), 'State_Location': 'MAHARASHTRA', 'Implementing_Agency': raw.get('Implementing_Agency', 'NHAI'),
            'Project_Tier': 'MAJOR', 'Original_Cost': orig_cost, 'Original_Duration_Months': orig_dur, 'Elapsed_Time_Months': elapsed,
            'Cumulative_Expenditure': cum_exp, 'Physical_Progress_Pct': phys_prog, 'Financial_Progress_Pct': (cum_exp / orig_cost) * 100.0,
            'Progress_Anomaly_Gap': ((cum_exp / orig_cost) * 100.0) - phys_prog, 'Expenditure_Burn_Rate': cum_exp / elapsed,
            'Actual_Physical_Run_Rate': phys_prog / elapsed, 'Expected_Physical_Run_Rate': 100.0 / orig_dur
        }
        
        df_in = pd.DataFrame([features]).reindex(columns=feature_columns, fill_value=0)
        for col, enc in encoders.items():
            if col in df_in.columns:
                df_in[col] = df_in[col].map({cls: idx for idx, cls in enumerate(enc.classes_)}).fillna(-1).astype(int)
                
        X_vals = df_in.astype(float).values
        c_pred = (models['Cost_Overrun_Pct']['model_a'].predict(X_vals)[0] + models['Cost_Overrun_Pct']['model_b'].predict(X_vals)[0]) / 2.0
        t_pred = (models['Time_Overrun_Months']['model_a'].predict(X_vals)[0] + models['Time_Overrun_Months']['model_b'].predict(X_vals)[0]) / 2.0
        
        shap_cost = explainer_cost.shap_values(X_vals)[0]
        shap_time = explainer_time.shap_values(X_vals)[0]
        c_drivers = sorted([{"feature": col, "impact": round(float(val), 2)} for col, val in zip(feature_columns, shap_cost)], key=lambda x: abs(x['impact']), reverse=True)[:3]
        t_drivers = sorted([{"feature": col, "impact": round(float(val), 2)} for col, val in zip(feature_columns, shap_time)], key=lambda x: abs(x['impact']), reverse=True)[:3]
        
        risk = "Critical" if t_pred >= 48 or c_pred >= 50 else "High" if t_pred >= 24 or c_pred >= 25 else "Medium" if t_pred >= 6 or c_pred >= 10 else "Low"
        return jsonify({"Final_Cost_Overrun": round(c_pred, 2), "Final_Time_Delay": round(t_pred, 2), "Risk_Tier": risk, "Top_Cost_Drivers": c_drivers, "Top_Time_Drivers": t_drivers})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/chat', methods=['POST'])
def chat_copilot():
    try:
        user_query = request.json.get('query', '').strip().lower()
        if len(user_query) <= 2 or any(g in user_query.split() for g in ['hi', 'hello', 'hey', 'help']):
            return jsonify({"response_html": "<p class='text-slate-100'>Hello Officer. Ask me to find the 'most delayed projects in Bihar' or 'Explain SHAP drivers for N04000050'.</p>"})

        shap_html = ""
        for proj_id in portfolio_df['Project_ID']:
            if str(proj_id).lower() in user_query:
                shap_html = get_project_shap_explanation(proj_id)
                break

        records, raw_title = search_rag_context(user_query)

        if not records and not shap_html:
            return jsonify({"response_html": "<p class='text-amber-400'>No records found. Try searching for 'critical risk', 'Bihar', or 'N04000050'.</p>"})

        response_html = shap_html
        if records:
            parts = raw_title.split('|')
            category_title = parts[0]
            summary_text = parts[1] if len(parts) > 1 else ""

            response_html += f"""
            <div class='mb-4'>
                <div class='text-xs font-semibold text-blue-400 uppercase tracking-wider'>{category_title}</div>
                <div class='text-sm text-slate-300 mt-2 bg-slate-950/40 p-3 rounded border border-slate-700/50'>{summary_text}</div>
            </div>
            <div class='overflow-x-auto rounded-xl border border-slate-700/80 bg-slate-950/60 mb-2'>
                <table class='w-full text-left text-xs text-slate-300'>
                    <thead class='bg-slate-800/80 text-slate-400 border-b border-slate-700/60'>
                        <tr><th class='p-3'>Project ID</th><th class='p-3'>State</th><th class='p-3'>Budget</th><th class='p-3'>Delay</th><th class='p-3'>Risk</th></tr>
                    </thead>
                    <tbody class='divide-y divide-slate-800/60'>
            """
            for r in records:
                rb = "text-rose-400 border-rose-500/30" if r.get('Risk_Tier') == 'Critical' else "text-amber-400 border-amber-500/30" if r.get('Risk_Tier') == 'High' else "text-emerald-400"
                response_html += f"<tr><td class='p-3 font-mono text-blue-400'>{r.get('Project_ID')}</td><td class='p-3'>{r.get('State_Location')}</td><td class='p-3'>₹{r.get('Original_Cost')} Cr</td><td class='p-3 text-amber-300'>{r.get('Predicted_Delay_Months')}m</td><td class='p-3'><span class='px-2 py-0.5 rounded-full border {rb}'>{r.get('Risk_Tier')}</span></td></tr>"
            response_html += "</tbody></table></div>"

        return jsonify({"response_html": response_html})
    except Exception as e:
        return jsonify({"response_html": f"<p class='text-rose-400'>Internal Error: {str(e)}</p>"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)