"""
Appli Streamlit - GAN Stability Lab (DCGAN vs WGAN-GP)

Se connecte au vrai backend FastAPI (pas de donnees simulees).
Lancer avec :  streamlit run frontend_app.py
Le backend doit deja tourner sur http://127.0.0.1:5000
"""

import base64
import io

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_BASE = "https://projet-annuel-93io.onrender.com"

st.set_page_config(
    page_title="GAN Stability Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Style — theme sombre "glassmorphism" avec halos degrades
# ---------------------------------------------------------------------------
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    html, body {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }
    .stApp {
        background:
            radial-gradient(1100px circle at 88% -8%, rgba(139, 92, 246, 0.35), transparent 55%),
            radial-gradient(900px circle at -8% 30%, rgba(34, 211, 238, 0.16), transparent 55%),
            radial-gradient(800px circle at 55% 105%, rgba(139, 92, 246, 0.18), transparent 60%),
            #0B0E1A;
        background-attachment: fixed;
    }
    .block-container {
        padding-top: 2.4rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    p, span, div, label { color: #E9EAF5; }
    /* ---------- Header ---------- */
    .app-title-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .app-logo {
        width: 44px;
        height: 44px;
        border-radius: 13px;
        background: linear-gradient(135deg, #8B5CF6 0%, #22D3EE 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        font-weight: 800;
        color: #0B0E1A;
        box-shadow: 0 0 24px rgba(139, 92, 246, 0.55), 0 4px 14px rgba(0, 0, 0, 0.4);
        flex-shrink: 0;
    }
    .app-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F5F6FF;
        margin: 0;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .app-subtitle {
        color: #9199B8;
        font-size: 0.94rem;
        margin: 0.2rem 0 1.9rem 0;
        padding-left: 3.35rem;
    }
    .app-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .app-badge .dot {
        width: 7px; height: 7px; border-radius: 50%;
    }
    .badge-ok { background: rgba(34, 211, 130, 0.12); color: #4ADE9C; }
    .badge-ok .dot { background: #4ADE9C; box-shadow: 0 0 8px #4ADE9C; }
    .badge-warn { background: rgba(248, 113, 113, 0.12); color: #F87171; }
    .badge-warn .dot { background: #F87171; box-shadow: 0 0 8px #F87171; }
    /* ---------- Glass cards ---------- */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-card, .card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        backdrop-filter: blur(18px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }
    .stat-card {
        padding: 1.1rem 1.2rem;
        transition: border-color 0.2s ease;
    }
    .stat-card:hover {
        border-color: rgba(139, 92, 246, 0.4);
    }
    .stat-label {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.76rem;
        font-weight: 600;
        color: #8890AC;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #F5F6FF;
        line-height: 1.2;
    }
    .stat-pill {
        display: inline-block;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .pill-green { background: rgba(74, 222, 156, 0.14); color: #4ADE9C; box-shadow: 0 0 14px rgba(74, 222, 156, 0.18) inset; }
    .pill-red { background: rgba(248, 113, 113, 0.14); color: #F87171; box-shadow: 0 0 14px rgba(248, 113, 113, 0.18) inset; }
    .pill-blue { background: rgba(96, 165, 250, 0.14); color: #60A5FA; box-shadow: 0 0 14px rgba(96, 165, 250, 0.18) inset; }
    .pill-gray { background: rgba(255, 255, 255, 0.08); color: #C3C8DE; }
    .card {
        padding: 1.35rem 1.45rem 1.55rem 1.45rem;
        height: 100%;
    }
    .card h4 {
        margin-top: 0;
        margin-bottom: 1.1rem;
        font-size: 1.02rem;
        font-weight: 700;
        color: #F5F6FF;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1224 0%, #0A0D1B 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] * {
        color: #E9EAF5 !important;
        font-family: 'Inter', sans-serif;
    }
    section[data-testid="stSidebar"] h3 {
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #7C86A8 !important;
        margin-bottom: 1.2rem;
    }
    section[data-testid="stSidebar"] label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #A8AFC8 !important;
    }
    section[data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%);
        color: white !important;
        border: none;
        border-radius: 11px;
        font-weight: 700;
        padding: 0.65rem 0;
        box-shadow: 0 0 22px rgba(139, 92, 246, 0.45);
        transition: box-shadow 0.2s ease;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.7);
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.12);
        border-radius: 11px;
    }
    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.6rem 1.25rem;
        font-weight: 600;
        color: #8890AC;
    }
    .stTabs [aria-selected="true"] {
        color: #C4B5FD !important;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.09);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path, **kwargs):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Impossible de joindre le backend distant (Render). Reessaie dans une minute (le service peut etre en train de se reveiller)."
    except requests.exceptions.Timeout:
        return None, "Le backend distant met du temps a repondre (reveil Render en cours). Reessaie dans 30-60 secondes."
    except requests.exceptions.HTTPError:
        return None, f"Erreur API ({r.status_code}) : {r.text}"
    except Exception as e:  # noqa: BLE001
        return None, f"Erreur inattendue : {e}"


def status_pill(status: str) -> str:
    """Transforme un statut brut backend en badge colore lisible."""
    raw = (status or "").lower()
    label = (status or "?").replace("_", " ").capitalize()
    if any(k in raw for k in ("complet", "termin", "done", "import")):
        css, label = "pill-green", "Termine" if "import" not in raw else "Importe"
    elif any(k in raw for k in ("fail", "error", "echec", "erreur")):
        css = "pill-red"
    elif any(k in raw for k in ("run", "cours", "pending", "attente")):
        css = "pill-blue"
    else:
        css = "pill-gray"
    return f'<span class="stat-pill {css}">{label}</span>'


def bool_pill(value: bool, true_label: str, false_label: str, invert=False) -> str:
    good = value if not invert else not value
    css = "pill-green" if good else "pill-red"
    label = true_label if value else false_label
    return f'<span class="stat-pill {css}">{label}</span>'


# ---------------------------------------------------------------------------
# Header + statut backend
# ---------------------------------------------------------------------------
health, health_err = api_get("/api/health")

col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown(
        '<div class="app-title-row">'
        '<div class="app-logo">GAN</div>'
        '<p class="app-title">GAN Stability Lab</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="app-subtitle">DCGAN vs WGAN-GP &middot; generation d\'echantillons '
        "et etude de stabilite d'entrainement</p>",
        unsafe_allow_html=True,
    )
with col_status:
    st.write("")
    if health_err:
        st.markdown(
            '<span class="app-badge badge-warn"><span class="dot"></span>Backend hors ligne</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="app-badge badge-ok"><span class="dot"></span>Backend connecte</span>',
            unsafe_allow_html=True,
        )

if health_err:
    st.error(health_err)
    st.stop()

runs, err = api_get("/api/runs")
if err:
    st.error(err)
    st.stop()

if not runs:
    st.warning(
        "Aucun run disponible pour le moment. Importe un modele entraine via "
        "`POST /api/runs/{run_id}/import` dans `/docs`, ou lance un entrainement avec `POST /api/runs`."
    )
    st.stop()

run_labels = {
    f"{r['model_type'].upper()} — seed {r['seed']}": r for r in runs
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controles")
    selected_label = st.selectbox("Modele / run", list(run_labels.keys()))
    n_samples = st.slider("Echantillons a generer", min_value=4, max_value=32, value=16, step=4)
    generate_clicked = st.button("Generer des echantillons", type="primary", use_container_width=True)
    if st.button("Actualiser", use_container_width=True):
        st.rerun()
    st.markdown("---")
    st.caption(f"{len(runs)} run(s) disponible(s) cote backend.")

selected_run = run_labels[selected_label]
run_id = selected_run["run_id"]

# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------
tab_gen, tab_stability = st.tabs(["Generation", "Etude de stabilite"])

with tab_gen:
    st.markdown(
        f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-label">Statut</div>
            <div class="stat-value">{status_pill(selected_run["status"])}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Epoques</div>
            <div class="stat-value">{selected_run['current_epoch']}/{selected_run['total_epochs']}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Converge</div>
            <div class="stat-value">{bool_pill(selected_run["converged"], "Oui", "Non")}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Mode collapse</div>
            <div class="stat-value">{bool_pill(selected_run["mode_collapse_detected"], "Oui", "Non", invert=True)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="card"><h4>Echantillons generes</h4>', unsafe_allow_html=True)
        if generate_clicked:
            with st.spinner("Generation en cours..."):
                data, err = api_get(f"/api/runs/{run_id}/samples", params={"n": n_samples})
            if err:
                st.error(err)
            else:
                img_bytes = base64.b64decode(data["image_base64"])
                img = Image.open(io.BytesIO(img_bytes))
                st.image(img, use_container_width=True)
        else:
            st.info("Choisis un modele et clique sur 'Generer des echantillons' dans la barre laterale.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card"><h4>Courbes de loss</h4>', unsafe_allow_html=True)
        losses, err = api_get(f"/api/runs/{run_id}/losses")
        if err:
            st.error(err)
        elif not losses:
            st.info(
                "Pas d'historique de loss pour ce run (normal pour un modele importe "
                "depuis un entrainement externe, ex. Colab)."
            )
        else:
            df = pd.DataFrame(losses)
            cols_to_plot = [c for c in ["g_loss", "d_loss"] if c in df.columns]
            st.line_chart(df.set_index("step")[cols_to_plot], height=280)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_stability:
    st.markdown('<div class="card"><h4>Tableau de stabilite &mdash; DCGAN vs WGAN-GP</h4>', unsafe_allow_html=True)
    stability, err = api_get("/api/stability")
    if err:
        st.error(err)
    elif not stability:
        st.info("Pas encore de donnees de stabilite. Lance des runs sur plusieurs seeds pour la remplir.")
    else:
        df_stab = pd.DataFrame(stability)
        if "n_converged" in df_stab.columns and "n_runs" in df_stab.columns:
            df_stab["taux_convergence_%"] = (df_stab["n_converged"] / df_stab["n_runs"] * 100).round(1)

        display_cols = [c for c in [
            "model_type", "dataset", "n_runs", "n_converged", "n_mode_collapse",
            "n_diverged", "taux_convergence_%", "avg_final_fid", "avg_wallclock_sec",
        ] if c in df_stab.columns]

        def _convergence_color(val):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v >= 80:
                return "background-color: rgba(74, 222, 156, 0.18); color: #4ADE9C; font-weight: 600;"
            if v >= 50:
                return "background-color: rgba(250, 204, 21, 0.18); color: #FACC15; font-weight: 600;"
            return "background-color: rgba(248, 113, 113, 0.18); color: #F87171; font-weight: 600;"

        styled = df_stab[display_cols].style
        if "taux_convergence_%" in display_cols:
            styled = styled.map(_convergence_color, subset=["taux_convergence_%"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        if "taux_convergence_%" in df_stab.columns:
            chart_df = df_stab.set_index("model_type")[["taux_convergence_%"]]
            st.bar_chart(chart_df, height=260)
    st.markdown("</div>", unsafe_allow_html=True)
