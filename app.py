# -*- coding: utf-8 -*-
"""
==============================================================================
 Application interactive - ASM Fouille de Donnees 2026 - Projet Ozone
 Binome : LEGUYADER & MOSTEFAOUI
==============================================================================
 Tableau de bord Streamlit permettant d'explorer, de tester et de modeliser
 la concentration en ozone a Rennes (ete 2001). L'utilisateur peut :
   - filtrer et trier le jeu de donnees,
   - choisir les variables et les parametres des analyses en temps reel,
   - relancer tests, ACP, regression, classification et arbre de decision.

 Lancement local :  streamlit run app.py
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# --------------------------------------------------------------------------
# Configuration generale
# --------------------------------------------------------------------------
st.set_page_config(page_title="Ozone - Fouille de donnees",
                   page_icon="🌫️", layout="wide")

QUANT = ["maxO3", "T9", "T12", "T15", "Ne9", "Ne12", "Ne15",
         "Vx9", "Vx12", "Vx15", "maxO3v"]
EXPLIC = ["T9", "T12", "T15", "Ne9", "Ne12", "Ne15",
          "Vx9", "Vx12", "Vx15", "maxO3v"]
SEED = 111


@st.cache_data
def charger():
    """Lit ozone.csv (separateur ';', decimale ','), prepare les variables."""
    ici = os.path.dirname(os.path.abspath(__file__))
    chemin = os.path.join(ici, "..", "data", "ozone.csv")
    if not os.path.exists(chemin):                 # repli si structure a plat
        chemin = os.path.join(ici, "ozone.csv")
    df = pd.read_csv(chemin, sep=";", decimal=",")
    df = df.drop(columns=["obs"])
    df["LmaxO3"] = np.log(df["maxO3"])
    df["LmaxO3v"] = np.log(df["maxO3v"])
    return df


ozone = charger()

# --------------------------------------------------------------------------
# Barre laterale : filtres globaux
# --------------------------------------------------------------------------
st.sidebar.title("🌫️ Projet Ozone")
st.sidebar.caption("ASM Fouille de Donnees 2026 — LEGUYADER & MOSTEFAOUI")
st.sidebar.markdown("---")
st.sidebar.subheader("Filtres globaux")

vents = sorted(ozone["vent"].unique())
sel_vent = st.sidebar.multiselect("Orientation du vent", vents, default=vents)
sel_pluie = st.sidebar.multiselect("Meteo", ["Sec", "Pluie"], default=["Sec", "Pluie"])
t_min, t_max = float(ozone["T12"].min()), float(ozone["T12"].max())
plage_t = st.sidebar.slider("Temperature a 12h (degC)", t_min, t_max, (t_min, t_max))

masque = (ozone["vent"].isin(sel_vent) & ozone["pluie"].isin(sel_pluie)
          & ozone["T12"].between(*plage_t))
data = ozone[masque].copy()
st.sidebar.metric("Jours selectionnes", f"{len(data)} / {len(ozone)}")
st.sidebar.markdown("---")
st.sidebar.info("Données : Air Breizh / Agrocampus Ouest — Rennes, été 2001.")

if len(data) < 10:
    st.warning("Filtres trop restrictifs : moins de 10 jours selectionnes. "
               "Les analyses sont desactivees, elargissez la selection.")
    st.stop()

# --------------------------------------------------------------------------
# En-tete
# --------------------------------------------------------------------------
st.title("Fouille de données : la pollution à l'ozone")
st.markdown("**Comprendre et prévoir le pic d'ozone journalier à partir de variables "
            "météorologiques** — Rennes, été 2001 (112 jours).")

onglets = st.tabs(["🏠 Données", "📊 Descriptif", "🧪 Tests",
                   "🧭 ACP", "📈 Régression & Prévision", "🌳 Classification"])

# ==========================================================================
# Onglet 1 : Donnees
# ==========================================================================
with onglets[0]:
    st.subheader("Le jeu de données")
    SEUIL_REG = 180   # seuil réglementaire d'information de la population (µg/m³)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", len(data))
    c2.metric("Pic O₃ moyen", f"{data['maxO3'].mean():.0f} µg/m³")
    c3.metric("Pic O₃ max", f"{data['maxO3'].max():.0f} µg/m³",
              help="Maximum observé à Rennes durant l'été 2001. "
                   f"Seuil réglementaire d'information de la population : {SEUIL_REG} µg/m³.")
    c4.metric("Jours de pluie", f"{(data['pluie']=='Pluie').sum()}")
    st.caption(
        f"📍 **Contexte du Pic O₃ max** — mesures réalisées à **Rennes** durant l'**été 2001**. "
        f"Le **seuil réglementaire d'information** de la population est fixé à **{SEUIL_REG} µg/m³** "
        f"(seuil d'alerte : 240 µg/m³). Sur l'ensemble de la période, ce seuil n'est **jamais "
        f"atteint** — le maximum observé est de **{int(ozone['maxO3'].max())} µg/m³**.")

    st.markdown("""
Chaque ligne est un **jour**. La variable cible est **maxO3** (pic d'ozone, µg/m³).
Les variables explicatives sont la **température** (T9/T12/T15), la **nébulosité**
(Ne9/Ne12/Ne15), la **composante Est-Ouest du vent** (Vx9/Vx12/Vx15), le **pic de
la veille** (maxO3v), l'**orientation du vent** et la **pluie**.
Le tableau ci-dessous est triable (clic sur l'en-tête) et filtrable via la barre latérale.
""")
    st.dataframe(data.round(2), use_container_width=True, height=330)
    st.download_button("⬇️ Télécharger la sélection (CSV)",
                       data.to_csv(index=False).encode("utf-8"),
                       "ozone_selection.csv", "text/csv")

# ==========================================================================
# Onglet 2 : Descriptif
# ==========================================================================
with onglets[1]:
    st.subheader("Statistiques descriptives")
    st.dataframe(data[QUANT].describe().T.round(2), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        v = st.selectbox("Variable à visualiser", QUANT, index=0)
        log = st.checkbox("Échelle logarithmique", value=(v in ["maxO3", "maxO3v"]))
        serie = np.log(data[v]) if log else data[v]
        fig = px.histogram(serie, nbins=20, marginal="box",
                           title=f"Distribution de {'log(' + v + ')' if log else v}",
                           color_discrete_sequence=["#1f6fb2"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        grp = st.selectbox("Comparer maxO3 selon", ["pluie", "vent"])
        fig = px.box(data, x=grp, y="maxO3", color=grp, points="all",
                     title=f"maxO3 selon {grp}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Matrice de corrélation (variables quantitatives)")
    fig = px.imshow(data[QUANT].corr().round(2), text_auto=True,
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# Onglet 3 : Tests
# ==========================================================================
with onglets[2]:
    st.subheader("Tests d'hypothèse")
    alpha = st.slider("Seuil de signification α", 0.01, 0.10, 0.05, 0.01)
    question = st.radio("Question scientifique", [
        "La pluie influence-t-elle le pic d'ozone ?",
        "L'orientation du vent influence-t-elle le pic d'ozone ?",
        "Le pic d'ozone diffère-t-il de celui de la veille ? (apparié)",
    ])

    if question.startswith("La pluie"):
        a = data.loc[data["pluie"] == "Sec", "LmaxO3"]
        b = data.loc[data["pluie"] == "Pluie", "LmaxO3"]
        lev = stats.levene(a, b)
        t = stats.ttest_ind(a, b, equal_var=(lev.pvalue >= alpha))
        st.write(f"**H₀** : la concentration moyenne (log) est identique les jours secs et pluvieux.")
        st.write(f"Test de Levene (égalité des variances) : p = {lev.pvalue:.4f} → "
                 f"test de {'Student' if lev.pvalue>=alpha else 'Welch'} retenu.")
        p = t.pvalue
        c1, c2 = st.columns(2)
        c1.metric("Moyenne jours secs", f"{np.exp(a.mean()):.1f} µg/m³")
        c2.metric("Moyenne jours de pluie", f"{np.exp(b.mean()):.1f} µg/m³")
    elif question.startswith("L'orientation"):
        groupes = [data.loc[data["vent"] == m, "LmaxO3"] for m in sorted(data["vent"].unique())]
        f = stats.f_oneway(*groupes)
        p = f.pvalue
        st.write("**H₀** : la concentration moyenne (log) est identique pour toutes les orientations de vent.")
        st.write("Outil : ANOVA à un facteur (test de Fisher).")
    else:
        t = stats.ttest_rel(data["LmaxO3"], data["LmaxO3v"])
        p = t.pvalue
        st.write("**H₀** : pas de différence moyenne entre le pic du jour et celui de la veille.")
        st.write("Outil : test de Student apparié.")

    decision = "❌ H₀ rejetée" if p < alpha else "✅ H₀ conservée"
    couleur = "red" if p < alpha else "green"
    st.markdown(f"### p-value = `{p:.2e}` → :{couleur}[{decision}] (α = {alpha})")

# ==========================================================================
# Onglet 4 : ACP
# ==========================================================================
with onglets[3]:
    st.subheader("Analyse en Composantes Principales (réduite)")
    Xs = StandardScaler().fit_transform(data[QUANT])
    pca = PCA().fit(Xs)
    scores = pca.transform(Xs)
    vr = pca.explained_variance_ratio_ * 100
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Éboulis des valeurs propres**")
        ax = np.arange(1, len(vr) + 1)
        fig = go.Figure()
        fig.add_bar(x=ax, y=vr, name="% variance", marker_color="#1f6fb2")
        fig.add_scatter(x=ax, y=np.cumsum(vr), name="% cumulé", line_color="#c0392b")
        fig.update_layout(xaxis_title="Composante", yaxis_title="% variance")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Les 2 premiers axes résument **{np.cumsum(vr)[1]:.1f} %** de l'information.")
    with col2:
        st.markdown("**Cercle des corrélations (plan 1-2)**")
        # Rendu matplotlib (carré, lisible) : le cercle reste circulaire quelle
        # que soit la largeur de la fenêtre, et les libellés ne se chevauchent pas.
        fig_c, axc = plt.subplots(figsize=(5.4, 5.4))
        axc.add_artist(plt.Circle((0, 0), 1, color="grey", fill=False, ls="--", lw=1))
        for i, v in enumerate(QUANT):
            coul = "#c0392b" if v == "maxO3" else "#1f6fb2"
            axc.annotate("", xy=(loadings[i, 0], loadings[i, 1]), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=coul, lw=1.6, alpha=0.85))
            axc.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, v,
                     color=coul, fontsize=9, ha="center", va="center",
                     fontweight="bold" if v == "maxO3" else "normal",
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.6))
        axc.axhline(0, color="grey", lw=0.6)
        axc.axvline(0, color="grey", lw=0.6)
        axc.set_xlim(-1.3, 1.3)
        axc.set_ylim(-1.3, 1.3)
        axc.set_aspect("equal")
        axc.set_xlabel(f"Dim 1 ({vr[0]:.1f} %)")
        axc.set_ylabel(f"Dim 2 ({vr[1]:.1f} %)")
        axc.tick_params(labelsize=8)
        st.pyplot(fig_c)
        st.caption("Deux flèches proches sont corrélées positivement ; opposées, négativement. "
                   "Plus une flèche est longue (proche du cercle), mieux la variable est représentée.")

    st.markdown("**Projection des jours** (la couleur représente une variable au choix)")
    col_coul = st.selectbox("Colorer les individus selon", ["maxO3", "pluie", "vent", "T12"])
    dfp = pd.DataFrame({"Dim1": scores[:, 0], "Dim2": scores[:, 1]})
    dfp[col_coul] = data[col_coul].values
    fig = px.scatter(dfp, x="Dim1", y="Dim2", color=col_coul,
                     color_continuous_scale="Turbo" if col_coul in ["maxO3", "T12"] else None,
                     labels={"Dim1": f"Dim 1 ({vr[0]:.1f} %)", "Dim2": f"Dim 2 ({vr[1]:.1f} %)"})
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# Onglet 5 : Regression et prevision
# ==========================================================================
with onglets[4]:
    st.subheader("Régression multiple et prévision de log(maxO3)")
    st.markdown("Sélectionnez les variables explicatives ; le modèle est ré-estimé en direct.")
    predicteurs = st.multiselect("Variables explicatives",
                                 ["T9", "T12", "T15", "Ne9", "Ne12", "Ne15",
                                  "Vx9", "Vx12", "Vx15", "LmaxO3v"],
                                 default=["T12", "Ne9", "Vx9", "LmaxO3v"])
    if predicteurs:
        modele = smf.ols("LmaxO3 ~ " + " + ".join(predicteurs), data=data).fit()
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{modele.rsquared:.3f}")
        c2.metric("R² ajusté", f"{modele.rsquared_adj:.3f}")
        c3.metric("AIC", f"{modele.aic:.1f}")

        tab = pd.DataFrame({"coefficient": modele.params.round(4),
                            "p-value": modele.pvalues.round(4)})
        tab["significatif (5%)"] = np.where(modele.pvalues < 0.05, "✔", "")
        st.dataframe(tab, use_container_width=True)

        pred_log = cross_val_predict(LinearRegression(), data[predicteurs], data["LmaxO3"],
                                     cv=KFold(10, shuffle=True, random_state=SEED))
        dfr = pd.DataFrame({"observé": data["maxO3"].values, "prédit": np.exp(pred_log)})
        fig = px.scatter(dfr, x="observé", y="prédit",
                         title="Pic d'ozone : observé vs prédit (validation croisée 10-plis)")
        lim = [dfr.min().min(), dfr.max().max()]
        fig.add_scatter(x=lim, y=lim, mode="lines", line_color="#c0392b", name="y = x")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sélectionnez au moins une variable explicative.")

# ==========================================================================
# Onglet 6 : Classification (CAH + arbre)
# ==========================================================================
with onglets[5]:
    st.subheader("Classification non supervisée et arbre de décision")

    st.markdown("#### 1. Classification Ascendante Hiérarchique (CAH, méthode de Ward)")
    k = st.slider("Nombre de classes (typologies de journées)", 2, 5, 3)
    Xs = StandardScaler().fit_transform(data[QUANT])
    Z = linkage(Xs, method="ward")
    classes = fcluster(Z, k, criterion="maxclust")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        dendrogram(Z, ax=ax, color_threshold=Z[-(k - 1), 2], no_labels=True)
        ax.axhline(Z[-(k - 1), 2], color="red", ls="--")
        ax.set_ylabel("Distance d'agrégation")
        st.pyplot(fig)
    with col2:
        prof = data.assign(classe=classes).groupby("classe")[
            ["maxO3", "T12", "Ne12", "maxO3v"]].mean().round(1)
        prof["effectif"] = pd.Series(classes).value_counts().sort_index().values
        st.markdown("**Profil moyen des classes**")
        st.dataframe(prof, use_container_width=True)

    st.markdown("#### 2. Arbre de décision : anticiper un épisode d'ozone élevé")
    seuil = st.slider("Seuil d'épisode élevé (µg/m³)", 100, 170, 120, 10)
    prof_max = st.slider("Profondeur maximale de l'arbre", 2, 4, 3)
    y = (data["maxO3"] >= seuil).astype(int)
    if y.sum() < 3 or y.sum() > len(y) - 3:
        st.warning("Seuil trop extrême pour cette sélection (classes déséquilibrées).")
    else:
        arbre = DecisionTreeClassifier(max_depth=prof_max, min_samples_leaf=5,
                                       random_state=SEED).fit(data[EXPLIC], y)
        st.caption(f"{int(y.sum())} jours « élevés » sur {len(y)} — "
                   f"précision sur l'échantillon : {arbre.score(data[EXPLIC], y)*100:.0f} %.")
        fig, ax = plt.subplots(figsize=(11, 5))
        plot_tree(arbre, feature_names=EXPLIC, class_names=["Normal", "Élevé"],
                  filled=True, rounded=True, impurity=False, fontsize=8, ax=ax)
        st.pyplot(fig)

st.markdown("---")
st.caption("Application réalisée sous Streamlit — Projet ASM Fouille de Données 2026, "
           "INP-ENSIACET. Binôme LEGUYADER & MOSTEFAOUI.")
