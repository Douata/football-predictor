import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

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
# CSS — CHAMPIONS LEAGUE STYLE (Bleu foncé & Or)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-deep:    #060d1a;
    --bg-primary: #0a1628;
    --bg-card:    #0d1f3c;
    --bg-card2:   #112447;
    --gold:       #c9a840;
    --gold-light: #e8c96a;
    --gold-dim:   #7a6320;
    --silver:     #8899bb;
    --white:      #eef2ff;
    --border:     #1e3a5f;
    --border-gold:#c9a84030;
    --text-muted: #5577aa;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--bg-primary) !important; }
.block-container { padding: 0 3rem 3rem !important; max-width: 1400px; }

div[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--white) !important;
    border-radius: 10px !important;
}
label[data-testid="stWidgetLabel"] p {
    color: var(--silver) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

.cl-header {
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-gold);
    padding: 2.5rem 0 2rem;
    text-align: center;
    margin: 0 -3rem 2rem;
}
.cl-stars { font-size: 0.65rem; letter-spacing: 8px; color: var(--gold); margin-bottom: 0.8rem; }
.cl-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.8rem; font-weight: 700;
    color: var(--white); letter-spacing: 6px;
    text-transform: uppercase; line-height: 1; margin: 0;
}
.cl-title span { color: var(--gold); }
.cl-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: var(--silver);
    letter-spacing: 3px; margin-top: 0.6rem;
}
.cl-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--border-gold); border: 1px solid var(--gold-dim);
    border-radius: 50px; padding: 0.35rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: var(--gold); letter-spacing: 2px; margin-top: 1rem;
}

.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.5rem;
}
.card-gold {
    background: var(--bg-card); border: 1px solid var(--border-gold);
    border-radius: 16px; padding: 1.5rem;
}
.card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; letter-spacing: 3px; color: var(--gold);
    text-transform: uppercase; margin-bottom: 1rem;
}

.match-container {
    background: var(--bg-primary); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.8rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
    margin: 1.5rem 0; position: relative; overflow: hidden;
}
.match-container::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}
.team-side { flex: 1; text-align: center; }
.team-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; letter-spacing: 3px; color: var(--silver);
    text-transform: uppercase; margin-bottom: 0.4rem;
}
.team-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem; font-weight: 700; color: var(--white);
    letter-spacing: 2px; text-transform: uppercase;
}
.vs-center { padding: 0 2rem; text-align: center; }
.vs-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem; font-weight: 700; color: var(--gold); letter-spacing: 3px;
}

.stButton > button {
    background: linear-gradient(135deg, #c9a840, #e8c96a, #c9a840) !important;
    color: #060d1a !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.2rem !important; font-weight: 700 !important;
    letter-spacing: 4px !important; text-transform: uppercase !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.75rem 2rem !important; width: 100% !important;
    box-shadow: 0 4px 24px rgba(201,168,64,0.35) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(201,168,64,0.55) !important;
}

.result-card {
    background: var(--bg-card); border: 1px solid var(--gold-dim);
    border-radius: 20px; padding: 2.5rem; text-align: center;
    position: relative; overflow: hidden;
    animation: revealResult 0.5s ease-out;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}
@keyframes revealResult {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 4px; color: var(--gold);
    text-transform: uppercase; margin-bottom: 0.6rem;
}
.result-icon { font-size: 2.5rem; margin-bottom: 0.4rem; }
.result-main {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem; font-weight: 700; color: var(--white);
    letter-spacing: 3px; text-transform: uppercase; line-height: 1.1;
}
.result-conf {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem; font-weight: 600; color: var(--gold); margin-top: 0.4rem;
}

.prob-row { display: flex; align-items: center; gap: 12px; margin: 0.7rem 0; }
.prob-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    color: var(--silver); width: 120px; flex-shrink: 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.prob-track {
    flex: 1; background: var(--bg-primary); border: 1px solid var(--border);
    border-radius: 50px; height: 8px; overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 50px; }
.prob-val {
    font-family: 'Rajdhani', sans-serif; font-size: 1rem; font-weight: 600;
    width: 50px; text-align: right; flex-shrink: 0;
}

.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 1rem; }
.stat-box {
    background: var(--bg-primary); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.8rem 0.5rem; text-align: center;
}
.stat-val {
    font-family: 'Rajdhani', sans-serif; font-size: 1.7rem;
    font-weight: 700; color: var(--white); line-height: 1;
}
.stat-lbl {
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
    letter-spacing: 1.5px; color: var(--text-muted);
    text-transform: uppercase; margin-top: 3px;
}

.form-section { text-align: center; margin-bottom: 0.8rem; }
.form-lbl {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    letter-spacing: 2px; color: var(--silver); text-transform: uppercase; margin-bottom: 8px;
}
.form-dots { display: flex; gap: 6px; justify-content: center; flex-wrap: wrap; }
.dot {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Rajdhani', sans-serif; font-size: 0.75rem; font-weight: 700;
}
.dot-W { background:#0f2e1a; border:1.5px solid #2a7a3a; color:#4aaa5a; }
.dot-D { background:#2a2010; border:1.5px solid var(--gold-dim); color:var(--gold); }
.dot-L { background:#2a0a0a; border:1.5px solid #7a2020; color:#cc4444; }

.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold-dim), transparent);
    margin: 1.5rem 0;
}

.model-bar {
    background: var(--bg-primary); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.7rem 1rem;
    display: flex; align-items: center; gap: 10px; margin-top: 1.2rem;
}
.model-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--gold);
    flex-shrink: 0; animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.model-txt {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: var(--silver); letter-spacing: 1px;
}
.model-txt span { color: var(--gold); }

.idle-state { text-align: center; padding: 4rem 0; }
.idle-icon { font-size: 4rem; opacity: 0.15; }
.idle-text {
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    letter-spacing: 3px; color: var(--text-muted);
    text-transform: uppercase; margin-top: 1rem;
}

.cl-footer {
    text-align: center; padding: 2rem 0 1rem;
    border-top: 1px solid var(--border); margin-top: 3rem;
}
.cl-footer-stars { color: var(--gold); font-size: 0.6rem; letter-spacing: 6px; }
.cl-footer-text {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    letter-spacing: 2px; color: var(--text-muted);
    text-transform: uppercase; margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    base   = os.path.dirname(os.path.abspath(__file__))
    model  = joblib.load(os.path.join(base, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(base, 'scaler.pkl'))
    df     = pd.read_csv(os.path.join(base, 'dataset_features.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    return model, scaler, df

try:
    model, scaler, df = load_assets()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

FEATURE_COLS = [
    'HST','AST','HC','AC','HTHG','HTAG',
    'Home_form','Away_form',
    'Home_avg_scored','Home_avg_conceded',
    'Away_avg_scored','Away_avg_conceded',
    'Form_diff','Avg_scored_diff','Avg_conceded_diff'
]

LEAGUES = {
    'PL': '🏴󠁧󠁢󠁥󠁮󠁧󠁿  Premier League',
    'SA': '🇮🇹  Serie A',
    'LL': '🇪🇸  La Liga'
}

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_teams(ligue):
    return sorted(set(
        df[df['Ligue']==ligue]['HomeTeam'].tolist() +
        df[df['Ligue']==ligue]['AwayTeam'].tolist()
    ))

def get_team_stats(team, n=5):
    matches = df[(df['HomeTeam']==team)|(df['AwayTeam']==team)].sort_values('Date').tail(n)
    if len(matches)==0: return None
    gs_l, gc_l, pts_l, form = [], [], [], []
    for _, m in matches.iterrows():
        h = m['HomeTeam']==team
        gs_l.append(m['FTHG'] if h else m['FTAG'])
        gc_l.append(m['FTAG'] if h else m['FTHG'])
        if m['FTR']==('H' if h else 'A'):   pts_l.append(3); form.append('W')
        elif m['FTR']=='D':                  pts_l.append(1); form.append('D')
        else:                                pts_l.append(0); form.append('L')
    return {
        'form': form, 'points': sum(pts_l),
        'avg_scored': round(np.mean(gs_l),1),
        'avg_conceded': round(np.mean(gc_l),1),
        'wins': form.count('W'), 'draws': form.count('D'), 'losses': form.count('L'),
        'last_match': matches.iloc[-1]['Date'].strftime('%d %b %Y'),
        'total': len(df[(df['HomeTeam']==team)|(df['AwayTeam']==team)])
    }

def form_pts(team, date, n=5):
    past = df[((df['HomeTeam']==team)|(df['AwayTeam']==team))&(df['Date']<date)].sort_values('Date').tail(n)
    pts = 0
    for _, m in past.iterrows():
        h = m['HomeTeam']==team
        if m['FTR']==('H' if h else 'A'): pts += 3
        elif m['FTR']=='D': pts += 1
    return pts

def avg_g(team, date, scored=True, n=5):
    past = df[((df['HomeTeam']==team)|(df['AwayTeam']==team))&(df['Date']<date)].sort_values('Date').tail(n)
    if len(past)==0: return 0
    g = []
    for _, m in past.iterrows():
        h = m['HomeTeam']==team
        g.append(m['FTHG'] if (h and scored) or (not h and not scored) else m['FTAG'])
    return round(np.mean(g),2)

def build_features(home, away):
    ref  = df['Date'].max() + pd.Timedelta(days=1)
    hf, af = form_pts(home,ref), form_pts(away,ref)
    hgs, hgc = avg_g(home,ref,True), avg_g(home,ref,False)
    ags, agc = avg_g(away,ref,True), avg_g(away,ref,False)
    hh = df[df['HomeTeam']==home].tail(5)
    ah = df[df['AwayTeam']==away].tail(5)
    def col_mean(frame, c): return frame[c].mean() if len(frame)>0 else df[c].mean()
    X = pd.DataFrame([[
        col_mean(hh,'HST'), col_mean(ah,'AST'),
        col_mean(hh,'HC'),  col_mean(ah,'AC'),
        col_mean(hh,'HTHG'),col_mean(ah,'HTAG'),
        hf, af, hgs, hgc, ags, agc,
        hf-af, hgs-ags, hgc-agc
    ]], columns=FEATURE_COLS)
    return scaler.transform(X)

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cl-header">
    <div class="cl-stars">★ ★ ★ ★ ★ ★ ★ ★</div>
    <h1 class="cl-title">Football <span>Predictor</span></h1>
    <div class="cl-subtitle">Intelligence Artificielle · Machine Learning · 3 Ligues Européennes</div>
    <div class="cl-badge">Modèle entraîné · 5 700 matchs · Accuracy 64.04 %</div>
</div>
""", unsafe_allow_html=True)

col_l, col_h, col_vs, col_a = st.columns([1.2, 2.2, 0.6, 2.2])

with col_l:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Championnat</div>', unsafe_allow_html=True)
    ligue = st.selectbox("Ligue", list(LEAGUES.keys()),
                         format_func=lambda x: LEAGUES[x],
                         label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

teams = get_teams(ligue)

with col_h:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Équipe Domicile</div>', unsafe_allow_html=True)
    home_team = st.selectbox("Domicile", teams,
                             label_visibility="collapsed", key="home")
    st.markdown('</div>', unsafe_allow_html=True)

with col_vs:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;
                height:100%;padding-top:0.5rem;">
        <div class="vs-text">VS</div>
    </div>""", unsafe_allow_html=True)

with col_a:
    away_opts = [t for t in teams if t != home_team]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Équipe Extérieure</div>', unsafe_allow_html=True)
    away_team = st.selectbox("Extérieur", away_opts,
                             label_visibility="collapsed", key="away")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="match-container">
    <div class="team-side">
        <div class="team-label">🏠 Domicile</div>
        <div class="team-name">{home_team}</div>
    </div>
    <div class="vs-center">
        <span style="font-size:1.5rem">⚽</span><br>
        <span class="vs-text">VS</span>
    </div>
    <div class="team-side">
        <div class="team-label">✈️ Extérieur</div>
        <div class="team-name">{away_team}</div>
    </div>
</div>
""", unsafe_allow_html=True)

_, col_btn, _ = st.columns([1.5, 2, 1.5])
with col_btn:
    predict = st.button("⚡  LANCER LA PRÉDICTION", use_container_width=True)

st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

if predict:
    with st.spinner("Analyse en cours..."):
        time.sleep(0.6)
        X_pred = build_features(home_team, away_team)
        proba  = model.predict_proba(X_pred)[0]
        pred   = model.predict(X_pred)[0]

    res_map = {
        0: (f"Victoire {away_team}", "✈️"),
        1: ("Match Nul",             "🤝"),
        2: (f"Victoire {home_team}", "🏆"),
    }
    label, icon = res_map[pred]
    conf = proba[pred] * 100

    col_pred, col_stats = st.columns([1, 1.1])

    with col_pred:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">◆ Prédiction du Modèle ◆</div>
            <div class="result-icon">{icon}</div>
            <div class="result-main">{label}</div>
            <div class="result-conf">{conf:.1f}% de confiance</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-gold" style="margin-top:1rem">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Probabilités</div>', unsafe_allow_html=True)
        bars = [
            (f"🏠 {home_team[:16]}", proba[2]*100, "#c9a840"),
            ("🤝 Match Nul",          proba[1]*100, "#5577aa"),
            (f"✈️ {away_team[:16]}", proba[0]*100, "#3a5580"),
        ]
        html_bars = ""
        for lbl, val, col in bars:
            html_bars += f"""
            <div class="prob-row">
              <div class="prob-label">{lbl}</div>
              <div class="prob-track">
                <div class="prob-fill" style="width:{val:.1f}%;background:{col}"></div>
              </div>
              <div class="prob-val" style="color:{col}">{val:.1f}%</div>
            </div>"""
        st.markdown(html_bars, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="model-bar">
            <div class="model-dot"></div>
            <div class="model-txt">
                Modèle : <span>Régression Logistique</span> &nbsp;·&nbsp;
                Accuracy : <span>64.04 %</span> &nbsp;·&nbsp;
                Base : <span>5 700 matchs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        for team, lbl, acc in [
            (home_team, f"🏠 Domicile — {home_team}", "#c9a840"),
            (away_team, f"✈️ Extérieur — {away_team}", "#5577aa")
        ]:
            s = get_team_stats(team)
            if not s: continue
            dots = "".join(
                f'<div class="dot dot-{r}">{r}</div>' for r in s['form']
            )
            st.markdown(f"""
            <div class="card" style="margin-bottom:1rem;border-top:2px solid {acc}30">
              <div class="card-title" style="color:{acc}">{lbl}</div>
              <div class="form-section">
                <div class="form-lbl">Forme récente — 5 derniers matchs</div>
                <div class="form-dots">{dots}</div>
              </div>
              <div class="stat-grid">
                <div class="stat-box">
                  <div class="stat-val" style="color:{acc}">{s['points']}/15</div>
                  <div class="stat-lbl">Points</div>
                </div>
                <div class="stat-box">
                  <div class="stat-val">{s['avg_scored']}</div>
                  <div class="stat-lbl">Buts/match</div>
                </div>
                <div class="stat-box">
                  <div class="stat-val">{s['avg_conceded']}</div>
                  <div class="stat-lbl">Encaissés</div>
                </div>
                <div class="stat-box">
                  <div class="stat-val" style="color:#4aaa5a">{s['wins']}</div>
                  <div class="stat-lbl">Victoires</div>
                </div>
                <div class="stat-box">
                  <div class="stat-val" style="color:{acc}">{s['draws']}</div>
                  <div class="stat-lbl">Nuls</div>
                </div>
                <div class="stat-box">
                  <div class="stat-val" style="color:#cc4444">{s['losses']}</div>
                  <div class="stat-lbl">Défaites</div>
                </div>
              </div>
              <div style="margin-top:0.8rem;text-align:center">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                             color:var(--text-muted);letter-spacing:1.5px">
                  Dernier match · {s['last_match']} · {s['total']} matchs dans la base
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="idle-state">
        <div class="idle-icon">🏆</div>
        <div class="idle-text">Sélectionne deux équipes et lance la prédiction</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="cl-footer">
    <div class="cl-footer-stars">★ ★ ★ ★ ★ ★ ★ ★</div>
    <div class="cl-footer-text">Football Predictor AI · INP-HB · Cours de Machine Learning · 2026</div>
</div>
""", unsafe_allow_html=True)