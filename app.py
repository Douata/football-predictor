import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Predictor AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — DESIGN SOMBRE PREMIUM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary:    #0a0e1a;
    --bg-card:       #111827;
    --bg-card2:      #1a2235;
    --accent-green:  #00ff88;
    --accent-blue:   #3b82f6;
    --accent-orange: #f59e0b;
    --accent-red:    #ef4444;
    --text-primary:  #f1f5f9;
    --text-muted:    #64748b;
    --border:        #1e293b;
    --glow-green:    0 0 20px rgba(0,255,136,0.3);
    --glow-blue:     0 0 20px rgba(59,130,246,0.3);
}

/* Reset global */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--bg-card) !important; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px; }
div[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 12px !important;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    letter-spacing: 6px;
    background: linear-gradient(135deg, #ffffff 0%, #00ff88 50%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin: 0;
    animation: titleGlow 3s ease-in-out infinite alternate;
}
@keyframes titleGlow {
    from { filter: drop-shadow(0 0 20px rgba(0,255,136,0.3)); }
    to   { filter: drop-shadow(0 0 40px rgba(59,130,246,0.5)); }
}
.hero-subtitle {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
    font-size: 0.85rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00ff8820, #3b82f620);
    border: 1px solid #00ff8840;
    border-radius: 50px;
    padding: 0.3rem 1.2rem;
    font-size: 0.75rem;
    color: var(--accent-green);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 2px;
    margin-top: 1rem;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-green), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.card:hover::before { opacity: 1; }
.card-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    letter-spacing: 3px;
    color: var(--text-muted);
    margin-bottom: 1.2rem;
    text-transform: uppercase;
}

/* ── VS Badge ── */
.vs-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    padding: 1.5rem 0;
}
.team-badge {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.5rem;
    text-align: center;
    flex: 1;
    min-width: 0;
}
.team-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 2px;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.team-label {
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: var(--text-muted);
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.vs-badge {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    color: var(--accent-green);
    text-shadow: var(--glow-green);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50%       { transform: scale(1.1); opacity: 0.7; }
}

/* ── Result Card ── */
.result-card {
    background: linear-gradient(135deg, #0f1923, #1a2a1a);
    border: 2px solid var(--accent-green);
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: var(--glow-green);
    animation: resultReveal 0.6s ease-out;
}
@keyframes resultReveal {
    from { opacity: 0; transform: translateY(20px) scale(0.95); }
    to   { opacity: 1; transform: translateY(0)   scale(1); }
}
.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 4px;
    color: var(--accent-green);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-main {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    letter-spacing: 3px;
    color: #ffffff;
    line-height: 1.1;
    text-shadow: 0 0 30px rgba(255,255,255,0.3);
}
.result-confidence {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    color: var(--accent-green);
    margin-top: 0.5rem;
    text-shadow: var(--glow-green);
}
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }

/* ── Probability Bars ── */
.prob-container { margin: 1.5rem 0; }
.prob-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0.8rem 0;
}
.prob-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    width: 130px;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    background: var(--bg-card2);
    border-radius: 50px;
    height: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.prob-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 1s ease-out;
}
.prob-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    width: 55px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Stat boxes ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}
.stat-box {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}
.stat-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--text-primary);
    line-height: 1;
}
.stat-lbl {
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}
.stat-accent { color: var(--accent-green) !important; }
.stat-accent-blue { color: var(--accent-blue) !important; }

/* ── Form dots ── */
.form-dots {
    display: flex;
    gap: 6px;
    justify-content: center;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}
.dot {
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.dot-W { background: #00ff8830; border: 2px solid var(--accent-green); color: var(--accent-green); }
.dot-D { background: #f59e0b20; border: 2px solid var(--accent-orange); color: var(--accent-orange); }
.dot-L { background: #ef444430; border: 2px solid var(--accent-red); color: var(--accent-red); }

/* ── Divider ── */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-green)40, var(--accent-blue)40, transparent);
    margin: 2rem 0;
}

/* ── Model badge ── */
.model-info {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.5rem;
}
.model-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 8px var(--accent-green);
    animation: blink 2s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
.model-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
}
.model-text span { color: var(--accent-green); }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #00ff88, #3b82f6) !important;
    color: #000 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.3rem !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(0,255,136,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,255,136,0.5) !important;
}

/* ── Selectbox labels ── */
label[data-testid="stWidgetLabel"] p {
    color: var(--text-muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 2px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES ET DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    """Charge le modèle, le scaler et le dataset enrichi."""
    base = os.path.dirname(os.path.abspath(__file__))

    model  = joblib.load(os.path.join(base, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(base, 'scaler.pkl'))
    df     = pd.read_csv(os.path.join(base, 'dataset_features.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    return model, scaler, df

try:
    model, scaler, df = load_model_and_data()
    model_ok = True
except Exception as e:
    model_ok = False
    st.error(f"⚠️ Erreur de chargement : {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'HST', 'AST', 'HC', 'AC', 'HTHG', 'HTAG',
    'Home_form', 'Away_form',
    'Home_avg_scored', 'Home_avg_conceded',
    'Away_avg_scored', 'Away_avg_conceded',
    'Form_diff', 'Avg_scored_diff', 'Avg_conceded_diff'
]

LEAGUES = {'PL': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League',
           'SA': '🇮🇹 Serie A',
           'LL': '🇪🇸 La Liga'}

def get_teams(ligue=None):
    if ligue:
        teams = sorted(set(
            df[df['Ligue'] == ligue]['HomeTeam'].tolist() +
            df[df['Ligue'] == ligue]['AwayTeam'].tolist()
        ))
    else:
        teams = sorted(set(df['HomeTeam'].tolist() + df['AwayTeam'].tolist()))
    return teams

def get_team_stats(team, n=5):
    """Calcule les stats récentes d'une équipe."""
    matches = df[
        (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
    ].sort_values('Date').tail(n)

    if len(matches) == 0:
        return None

    goals_scored, goals_conceded, points = [], [], []
    form_results = []

    for _, m in matches.iterrows():
        is_home = m['HomeTeam'] == team
        gs = m['FTHG'] if is_home else m['FTAG']
        gc = m['FTAG'] if is_home else m['FTHG']
        goals_scored.append(gs)
        goals_conceded.append(gc)
        if m['FTR'] == ('H' if is_home else 'A'):
            points.append(3); form_results.append('W')
        elif m['FTR'] == 'D':
            points.append(1); form_results.append('D')
        else:
            points.append(0); form_results.append('L')

    return {
        'form'           : form_results,
        'points'         : sum(points),
        'avg_scored'     : round(np.mean(goals_scored), 2),
        'avg_conceded'   : round(np.mean(goals_conceded), 2),
        'wins'           : form_results.count('W'),
        'draws'          : form_results.count('D'),
        'losses'         : form_results.count('L'),
        'last_match'     : matches.iloc[-1]['Date'].strftime('%d %b %Y'),
        'total_matches'  : len(df[(df['HomeTeam']==team)|(df['AwayTeam']==team)])
    }

def get_recent_form_value(team, date, n=5):
    past = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
        (df['Date'] < date)
    ].sort_values('Date').tail(n)
    pts = 0
    for _, m in past.iterrows():
        is_home = m['HomeTeam'] == team
        if m['FTR'] == ('H' if is_home else 'A'): pts += 3
        elif m['FTR'] == 'D': pts += 1
    return pts

def get_avg_goals(team, date, scored=True, n=5):
    past = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
        (df['Date'] < date)
    ].sort_values('Date').tail(n)
    if len(past) == 0: return 0
    goals = []
    for _, m in past.iterrows():
        is_home = m['HomeTeam'] == team
        if scored:
            goals.append(m['FTHG'] if is_home else m['FTAG'])
        else:
            goals.append(m['FTAG'] if is_home else m['FTHG'])
    return round(np.mean(goals), 2)

def build_features(home_team, away_team):
    """Construit les features pour la prédiction."""
    ref_date = df['Date'].max() + pd.Timedelta(days=1)

    home_form    = get_recent_form_value(home_team, ref_date)
    away_form    = get_recent_form_value(away_team, ref_date)
    home_scored  = get_avg_goals(home_team, ref_date, scored=True)
    home_conc    = get_avg_goals(home_team, ref_date, scored=False)
    away_scored  = get_avg_goals(away_team, ref_date, scored=True)
    away_conc    = get_avg_goals(away_team, ref_date, scored=False)

    # Moyennes historiques pour HST, AST, HC, AC, HTHG, HTAG
    home_hist = df[df['HomeTeam'] == home_team].tail(5)
    away_hist = df[df['AwayTeam'] == away_team].tail(5)

    hst  = home_hist['HST'].mean()  if len(home_hist) > 0 else df['HST'].mean()
    ast  = away_hist['AST'].mean()  if len(away_hist) > 0 else df['AST'].mean()
    hc   = home_hist['HC'].mean()   if len(home_hist) > 0 else df['HC'].mean()
    ac   = away_hist['AC'].mean()   if len(away_hist) > 0 else df['AC'].mean()
    hthg = home_hist['HTHG'].mean() if len(home_hist) > 0 else df['HTHG'].mean()
    htag = away_hist['HTAG'].mean() if len(away_hist) > 0 else df['HTAG'].mean()

    features = pd.DataFrame([[
        hst, ast, hc, ac, hthg, htag,
        home_form, away_form,
        home_scored, home_conc,
        away_scored, away_conc,
        home_form - away_form,
        home_scored - away_scored,
        home_conc - away_conc
    ]], columns=FEATURE_COLS)

    return scaler.transform(features)

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-subtitle">Intelligence Artificielle · Machine Learning</p>
    <h1 class="hero-title">⚽ FOOTBALL PREDICTOR</h1>
    <div class="hero-badge">◆ MODÈLE ENTRAÎNÉ · 5700 MATCHS · 3 LIGUES EUROPÉENNES ◆</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# ── SÉLECTION LIGUE + ÉQUIPES ─────────────────────────────────────────────────
col_l, col_h, col_vs, col_a = st.columns([1.2, 2, 0.8, 2])

with col_l:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">🏆 Ligue</p>', unsafe_allow_html=True)
    ligue_choice = st.selectbox(
        "Championnat",
        options=list(LEAGUES.keys()),
        format_func=lambda x: LEAGUES[x],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

teams = get_teams(ligue_choice)

with col_h:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">🏠 Équipe Domicile</p>', unsafe_allow_html=True)
    home_team = st.selectbox("Domicile", options=teams, label_visibility="collapsed", key="home")
    st.markdown('</div>', unsafe_allow_html=True)

with col_vs:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;height:100%;padding-top:1rem;">
        <div class="vs-badge">VS</div>
    </div>
    """, unsafe_allow_html=True)

with col_a:
    away_options = [t for t in teams if t != home_team]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">✈️ Équipe Extérieure</p>', unsafe_allow_html=True)
    away_team = st.selectbox("Extérieur", options=away_options, label_visibility="collapsed", key="away")
    st.markdown('</div>', unsafe_allow_html=True)

# ── AFFICHAGE DU MATCH ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="vs-container">
    <div class="team-badge">
        <div class="team-label">🏠 Domicile</div>
        <div class="team-name">{home_team}</div>
    </div>
    <div class="vs-badge">⚽</div>
    <div class="team-badge">
        <div class="team-label">✈️ Extérieur</div>
        <div class="team-name">{away_team}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── BOUTON PRÉDIRE ────────────────────────────────────────────────────────────
col_btn = st.columns([1, 2, 1])[1]
with col_btn:
    predict_btn = st.button("⚡ LANCER LA PRÉDICTION", use_container_width=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RÉSULTAT + STATS
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:

    # Animation de chargement
    with st.spinner("🧠 Analyse en cours..."):
        time.sleep(0.8)
        X_pred = build_features(home_team, away_team)
        proba  = model.predict_proba(X_pred)[0]
        pred   = model.predict(X_pred)[0]

    # Mapping résultat
    result_map = {
        0: {"label": f"Victoire {away_team}",  "emoji": "✈️",  "color": "#3b82f6"},
        1: {"label": "Match Nul",               "emoji": "🤝",  "color": "#f59e0b"},
        2: {"label": f"Victoire {home_team}",   "emoji": "🏠",  "color": "#00ff88"},
    }
    res = result_map[pred]

    # ── RÉSULTAT PRINCIPAL ───────────────────────────────────────────────────
    col_res, col_stats = st.columns([1, 1])

    with col_res:
        confidence = proba[pred] * 100
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">◆ Prédiction du Modèle ◆</div>
            <div class="result-emoji">{res['emoji']}</div>
            <div class="result-main">{res['label']}</div>
            <div class="result-confidence">{confidence:.1f}% de confiance</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Barres de probabilité ────────────────────────────────────────────
        st.markdown('<div class="card" style="margin-top:1rem">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">📊 Probabilités</p>', unsafe_allow_html=True)

        prob_data = [
            {"label": f"🏠 {home_team[:18]}", "val": proba[2]*100, "color": "#00ff88"},
            {"label": "🤝 Match Nul",          "val": proba[1]*100, "color": "#f59e0b"},
            {"label": f"✈️ {away_team[:18]}", "val": proba[0]*100, "color": "#3b82f6"},
        ]

        bars_html = '<div class="prob-container">'
        for p in prob_data:
            bars_html += f"""
            <div class="prob-row">
                <div class="prob-label">{p['label']}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill"
                         style="width:{p['val']:.1f}%;
                                background:linear-gradient(90deg,{p['color']}80,{p['color']});">
                    </div>
                </div>
                <div class="prob-val" style="color:{p['color']}">{p['val']:.1f}%</div>
            </div>"""
        bars_html += '</div>'
        st.markdown(bars_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Info modèle ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="model-info">
            <div class="model-dot"></div>
            <div class="model-text">
                Modèle : <span>Régression Logistique</span> ·
                Accuracy : <span>64.04%</span> ·
                Données : <span>5 700 matchs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── STATS DES ÉQUIPES ────────────────────────────────────────────────────
    with col_stats:
        for team, label, color in [
            (home_team, "🏠 Domicile", "#00ff88"),
            (away_team, "✈️ Extérieur", "#3b82f6")
        ]:
            stats = get_team_stats(team)
            if stats:
                # Dots de forme
                dots_html = '<div class="form-dots">'
                for r in stats['form']:
                    cls = {'W': 'dot-W', 'D': 'dot-D', 'L': 'dot-L'}[r]
                    dots_html += f'<div class="dot {cls}">{r}</div>'
                dots_html += '</div>'

                st.markdown(f"""
                <div class="card">
                    <p class="card-title" style="color:{color}">{label} — {team}</p>
                    <div style="text-align:center;margin-bottom:0.5rem;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                     color:var(--text-muted);letter-spacing:2px;">
                            FORME RÉCENTE (5 DERNIERS MATCHS)
                        </span>
                        {dots_html}
                    </div>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-val stat-accent">{stats['points']}/15</div>
                            <div class="stat-lbl">Points</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val">{stats['avg_scored']}</div>
                            <div class="stat-lbl">Buts/match</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val">{stats['avg_conceded']}</div>
                            <div class="stat-lbl">Encaissés</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val" style="color:#00ff88">{stats['wins']}</div>
                            <div class="stat-lbl">Victoires</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val" style="color:#f59e0b">{stats['draws']}</div>
                            <div class="stat-lbl">Nuls</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val" style="color:#ef4444">{stats['losses']}</div>
                            <div class="stat-lbl">Défaites</div>
                        </div>
                    </div>
                    <div style="margin-top:1rem;text-align:center;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                     color:var(--text-muted);">
                            DERNIER MATCH · {stats['last_match']} ·
                            {stats['total_matches']} MATCHS DANS LA BASE
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    # État initial — invitation à prédire
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;opacity:0.4;">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:5rem;color:#00ff88;">⚽</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                    letter-spacing:3px;color:#64748b;text-transform:uppercase;">
            Sélectionne deux équipes et lance la prédiction
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ◆ FOOTBALL PREDICTOR AI · INP-HB · COURS DE MACHINE LEARNING · 2026 ◆
</div>
""", unsafe_allow_html=True)
