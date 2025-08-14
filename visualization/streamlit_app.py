import os
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =========================
# Theme & Palette
# =========================

st.set_page_config(page_title="UWE Bristol Dashboard", layout="wide")
PX_TEMPLATE = "plotly_white"
# Color palette
COLOR_UWE = "#0072B2"  
# UWE name on board 
UWE_NAME = "UWE Bristol"
          # UWE color sc
COLOR_COMPETITORS = [             # distinct, professional competitor colors
    "#6C757D",  # slate
    "#808000",  # olive
    "#800000",  # burgundy
    "#6A5ACD",  # muted purple
    "#A0522D",  # sienna
    "#9467BD",  # violet
    "#8C564B"   # brown
]
COLOR_HIGHLIGHT = "#E69F00"       # annotations/callouts
COLOR_POSITIVE = "#009E73"        # helpful (green)
COLOR_NEGATIVE = "#D55E00"        # harmful (red/orange)
TEXT_DARK = "#111827"

# colorblind‑safe list for subjects (Okabe–Ito + additions)
SUBJECT_PALETTE = [
    "#0072B2","#E69F00","#009E73","#D55E00","#CC79A7",
    "#56B4E9","#F0E442","#7F7F7F","#332288","#88CCEE",
    "#44AA99","#117733","#999933","#DDCC77","#661100"
]

FILE_PATH = "Guardian_Cleaned_Final.xlsx"
RECENT_WINDOW_SIZE = 5

# =========================
# Utilities
# =========================
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    for c in ["Institution", "Subject"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    num_cols = [
        "Rank","Guardian Score","Satisfied with Course","Satisfied with Teaching",
        "Satisfied with Feedback","Student to Staff Ratio","Spend per Student",
        "Average Entry Tariff","Value Added Score","Career after 15 months","Continuation"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data
def load_sheet(name: str) -> pd.DataFrame:
    if not os.path.exists(FILE_PATH):
        st.error(f"Data file not found: {os.path.abspath(FILE_PATH)}")
        st.stop()

    try:
        return clean(pd.read_excel(FILE_PATH, sheet_name=name))
    except Exception as e:
        st.error(f"Could not open sheet '{name}': {e}")
        st.stop()


def slice_years(df: pd.DataFrame, mode: str):
    years = sorted(df["Year"].dropna().unique().tolist())
    if not years:
        return df.iloc[0:0].copy(), []
    dom = years if mode == "Full period" else years[-RECENT_WINDOW_SIZE:]
    return df[df["Year"].isin(dom)].copy(), dom

def fixed_color_map(institutions):
    cmap = {UWE_NAME: COLOR_UWE}
    j = 0
    for inst in institutions:
        if inst == UWE_NAME:
            continue
        cmap[inst] = COLOR_COMPETITORS[j % len(COLOR_COMPETITORS)]
        j += 1
    return cmap

def emphasize_uwe(fig: go.Figure, color_map: dict) -> go.Figure:
    # base styling + consistent colors
    for i, tr in enumerate(fig.data):
        name = getattr(tr, "name", "")
        col = color_map.get(name, "#9AA0A6")
        fig.data[i].update(line=dict(color=col, width=2.3),
                           marker=dict(size=7),
                           opacity=0.9)
    # boost UWE
    for i, tr in enumerate(fig.data):
        if getattr(tr, "name", "") == UWE_NAME:
            fig.data[i].update(line=dict(color=COLOR_UWE, width=4.8),
                               marker=dict(size=10),
                               opacity=1.0)
    # bring UWE to top
    idx = next((i for i, tr in enumerate(fig.data) if getattr(tr, "name", "") == UWE_NAME), None)
    if idx not in (None, 0):
        d = list(fig.data); uwe = d.pop(idx); d.insert(0, uwe); fig.data = tuple(d)
    fig.update_layout(template=PX_TEMPLATE, legend_title_text="")
    return fig

def lin_trend(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return None, None
    b1, b0 = np.polyfit(x, y, 1)  # y = b1*x + b0
    return b1, b0

def corr_with_rank(df, cols):
    use = df[["Rank"] + cols].apply(pd.to_numeric, errors="coerce").dropna()
    if use.empty:
        return pd.Series(dtype=float)
    return use.corr(numeric_only=True)["Rank"].drop("Rank").sort_values()

def load_logo_base64(path="uwe_logo.png"):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# =========================
# Banner
# =========================
logo_b64 = load_logo_base64("uwe_logo.png")
banner_html = f"""
<style>
.reportview-container .main .block-container {{ padding-top: 0rem; }}
h1, h2, h3, h4, h5, h6, .stMetric label {{ color: {TEXT_DARK}; }}
.uwe-banner {{
  width: 100%;
  background: #f5f7fb;
  color: {TEXT_DARK};
  padding: 20px 24px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 14px;
  border: 1px solid #e5e7eb;
}}
.uwe-title {{ font-size: 30px; font-weight: 800; line-height: 1.1; margin: 0; color: {COLOR_UWE}; }}
.uwe-sub {{ font-size: 14px; opacity: 0.9; margin: 2px 0 0 0; }}
.uwe-logo {{ height: 42px; width: auto; border-radius: 6px; }}
</style>
<div class="uwe-banner">
  {"<img class='uwe-logo' src='data:image/png;base64," + logo_b64 + "'/>" if logo_b64 else ""}
  <div>
    <div class="uwe-title">UWE Bristol Dashboard — Admissions & Marketing</div>
    <div class="uwe-sub">Guardian University Guide • Full-period or Recent 5-year views • UWE highlighted</div>
  </div>
</div>
"""
st.markdown(banner_html, unsafe_allow_html=True)

# =========================
# Load data
# =========================
inst = load_sheet("Institution Level Data")
subj = load_sheet("Subject Level Data")

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Controls")
year_mode = st.sidebar.radio("Time range", ["Full period", "Recent 5 years"], index=0)
inst_sliced, inst_years = slice_years(inst, year_mode)
if not inst_years:
    st.error("No valid years found in Institution data.")
    st.stop()

inst_names_all = sorted(inst["Institution"].dropna().unique().tolist())
defaults = [n for n in [UWE_NAME, "Bristol", "Bath", "Bath Spa", "Cardiff", "Cardiff Met", "Nottingham Trent"] if n in inst_names_all][:4]
if not defaults:
    defaults = inst_names_all[:4]

chosen_insts = st.sidebar.multiselect("Institutions", options=inst_names_all, default=defaults)
focus_year = st.sidebar.slider(
    "Focus Year",
    min_value=int(min(inst_years)), max_value=int(max(inst_years)),
    value=int(max(inst_years)), step=1
)

subj_uwe = subj[subj["Institution"] == UWE_NAME].copy()
subj_sliced, subj_years = slice_years(subj_uwe, year_mode)

COLOR_MAP = fixed_color_map(chosen_insts)

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["🏁 Overview", "📚 Subjects", "❓ Answers (Q1–Q6)"])

# -----------------------------------------
# TAB 1: Overview
# -----------------------------------------
with tab1:
    # KPI strip
    u_all = inst_sliced[inst_sliced["Institution"] == UWE_NAME]
    u_cur = u_all[u_all["Year"] == focus_year]
    k = st.columns(4)
    k[0].metric("Current Rank", int(u_cur["Rank"].iloc[0]) if not u_cur.empty and pd.notna(u_cur["Rank"].iloc[0]) else "—")
    k[1].metric("Guardian Score", f"{float(u_cur['Guardian Score'].iloc[0]):.1f}" if not u_cur.empty and pd.notna(u_cur["Guardian Score"].iloc[0]) else "—")
    k[2].metric("Career after 15 months", f"{float(u_cur['Career after 15 months'].iloc[0]):.1f}%" if not u_cur.empty and pd.notna(u_cur['Career after 15 months'].iloc[0]) else "—")
    # Important driver (correlation)
    factor_cols_kpi = [c for c in ["Career after 15 months","Satisfied with Course","Satisfied with Teaching","Student to Staff Ratio"] if c in inst_sliced.columns]
    imp_series = corr_with_rank(inst_sliced, factor_cols_kpi)
    k[3].metric("Top driver (corr→rank)", imp_series.index[0] if isinstance(imp_series, pd.Series) and not imp_series.empty else "—")

    st.markdown("### Satisfaction (line charts)")
    c1, c2 = st.columns([1.1, 1.9])
    uwe_year = inst_sliced[(inst_sliced["Institution"] == UWE_NAME) & (inst_sliced["Year"] == focus_year)]
    uwe_ts = inst_sliced[inst_sliced["Institution"] == UWE_NAME].sort_values("Year")
    sat_cols_all = ["Satisfied with Course","Satisfied with Teaching","Satisfied with Feedback"]
    sat_cols = [c for c in sat_cols_all if c in inst_sliced.columns]

    if sat_cols and not uwe_ts.empty:
        if not uwe_year.empty:
            k1,k2,k3 = c1.columns(3)
            if "Satisfied with Course" in sat_cols:  k1.metric("Course",  f"{uwe_year['Satisfied with Course'].iloc[0]:.1f}%")
            if "Satisfied with Teaching" in sat_cols: k2.metric("Teaching",f"{uwe_year['Satisfied with Teaching'].iloc[0]:.1f}%")
            if "Satisfied with Feedback" in sat_cols: k3.metric("Feedback",f"{uwe_year['Satisfied with Feedback'].iloc[0]:.1f}%")

        fig_sat_lines = px.line(
            uwe_ts, x="Year", y=sat_cols, markers=True, template=PX_TEMPLATE,
            title="Student Satisfaction — UWE over time",
            color_discrete_map={
                "Satisfied with Course": COLOR_UWE,
                "Satisfied with Teaching": "#66A9CF",
                "Satisfied with Feedback": "#A6CEE3"
            }
        )
        fig_sat_lines.update_traces(line=dict(width=3.2), marker=dict(size=8))
        c2.plotly_chart(fig_sat_lines, use_container_width=True)
    else:
        st.info("Satisfaction time‑series not available.")

    st.markdown("### Rank vs Guardian Score (clarified axes & styles)")
    uwe_series = inst_sliced[inst_sliced["Institution"] == UWE_NAME].sort_values("Year")
    if not uwe_series.empty:
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(
            x=uwe_series["Year"], y=uwe_series["Rank"], name="Rank (lower is better)",
            mode="lines+markers", line=dict(color=COLOR_UWE, width=4.8), marker=dict(size=10)
        ))
        fig_dual.add_trace(go.Scatter(
            x=uwe_series["Year"], y=uwe_series["Guardian Score"], name="Guardian Score (higher is better)",
            mode="lines+markers", yaxis="y2",
            line=dict(color="#333333", width=2.2, dash="dot"), marker=dict(size=7)
        ))
        fig_dual.update_layout(
            template=PX_TEMPLATE, title="UWE — Rank vs Guardian Score",
            yaxis=dict(title="Rank (lower is better)", autorange="reversed"),
            yaxis2=dict(title="Guardian Score", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    st.markdown("### Comparison — Rank Over Time (UWE vs competitors)")
    comp_df = inst_sliced[inst_sliced["Institution"].isin(chosen_insts)].sort_values(["Institution","Year"])
    if comp_df.empty:
        st.info("No data for selected institutions.")
    else:
        fig_cmp = px.line(
            comp_df, x="Year", y="Rank", color="Institution", markers=True,
            template=PX_TEMPLATE, color_discrete_map=COLOR_MAP
        )
        fig_cmp.update_yaxes(autorange="reversed", title_text="Rank (lower is better)")
        fig_cmp.update_xaxes(title_text="Year")
        fig_cmp = emphasize_uwe(fig_cmp, COLOR_MAP)
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("### Competitor Analysis — Employability (Career after 15 months)")
    inst_year_list = sorted(inst_sliced["Year"].dropna().unique().tolist())
    years_emp = [y for y in range(2018, 2023) if y in inst_year_list]
    default_shortlist = [u for u in ["Nottingham Trent","Portsmouth","Oxford Brookes","Cardiff","Plymouth","Gloucestershire","Bournemouth","Bath Spa","Brighton"] if u in inst_names_all][:5]
    shortlist = st.multiselect("Shortlist competitors", options=[i for i in inst_names_all if i != UWE_NAME],
                               default=default_shortlist, key="emp_shortlist")
    if years_emp and shortlist:
        fig_pies = make_subplots(rows=1, cols=len(years_emp), specs=[[{"type":"domain"}]*len(years_emp)],
                                 subplot_titles=[str(y) for y in years_emp])
        for ci, yr in enumerate(years_emp, start=1):
            dfy = inst_sliced[inst_sliced["Year"] == yr]
            group = dfy[dfy["Institution"].isin([UWE_NAME] + shortlist)].copy()
            if group.empty or "Career after 15 months" not in group.columns:
                labels, values, colors = [UWE_NAME], [0], [COLOR_UWE]
            else:
                labels = group["Institution"].tolist()
                values = group["Career after 15 months"].tolist()
                vals = np.array(values, dtype=float)
                vals = np.maximum(vals, 0)
                if vals.sum() == 0:
                    vals = np.ones_like(vals)
                colors = [COLOR_UWE if n == UWE_NAME else "#B8BCC2" for n in labels]
                values = vals
            fig_pies.add_trace(
                go.Pie(labels=labels, values=values, name=str(yr),
                       hole=0.45, textinfo="percent", insidetextorientation="auto",
                       marker=dict(colors=colors)),
                row=1, col=ci
            )
        fig_pies.update_layout(template=PX_TEMPLATE, showlegend=False, height=300)
        st.plotly_chart(fig_pies, use_container_width=True)
        st.caption("Share within selected group each year, using 'Career after 15 months' as the weight. UWE slice shown in bold blue.")

# -----------------------------------------
# TAB 2: Subjects
# -----------------------------------------
with tab2:
    st.markdown("### Top Subjects — Five‑year Trend (Guardian Score and optional Rank)")
    if subj_sliced.empty:
        st.info("No UWE subject data in this time range.")
    else:
        latest_subj_year = int(subj_sliced["Year"].max())
        base_latest = subj_sliced[subj_sliced["Year"] == latest_subj_year].dropna(subset=["Guardian Score"])
        if base_latest.empty:
            st.info("No Guardian Score in latest year for subjects.")
        else:
            top_n_choice = st.radio("How many subjects?", ["Top 5", "Top 10"], horizontal=True)
            top_n = 5 if top_n_choice == "Top 5" else 10
            top_subjects = base_latest.nlargest(top_n, "Guardian Score")["Subject"].tolist()

            all_years_sorted = sorted(subj_sliced["Year"].dropna().unique().tolist())
            last5_years = all_years_sorted[-5:] if len(all_years_sorted) >= 5 else all_years_sorted

            trend_df = subj_sliced[subj_sliced["Subject"].isin(top_subjects) & subj_sliced["Year"].isin(last5_years)].copy()
            trend_df = trend_df.sort_values(["Subject","Year"])

            fig_ts_score = px.line(
                trend_df, x="Year", y="Guardian Score", color="Subject", markers=True,
                template=PX_TEMPLATE, title=f"Top {top_n} Subjects — Guardian Score Trend (last 5 years)",
                color_discrete_sequence=SUBJECT_PALETTE
            )
            fig_ts_score.update_traces(line=dict(width=3.2), marker=dict(size=7))
            st.plotly_chart(fig_ts_score, use_container_width=True)

            show_rank = st.checkbox("Show Rank trend for the same subjects (up = better rank)", value=True)
            if show_rank and "Rank" in trend_df.columns:
                fig_ts_rank = px.line(
                    trend_df, x="Year", y="Rank", color="Subject", markers=True,
                    template=PX_TEMPLATE, title=f"Top {top_n} Subjects — Rank Trend (last 5 years, lower is better)",
                    color_discrete_sequence=SUBJECT_PALETTE
                )
                fig_ts_rank.update_traces(line=dict(width=3.2), marker=dict(size=7))
                fig_ts_rank.update_yaxes(autorange="reversed", title_text="Rank (lower is better)")
                st.plotly_chart(fig_ts_rank, use_container_width=True)

    st.markdown("### Top 10 in latest year (easy view)")
    if not subj_sliced.empty and "Guardian Score" in subj_sliced.columns:
        ly = int(max(subj_years))
        ls = subj_sliced[subj_sliced["Year"] == ly].dropna(subset=["Guardian Score"]).copy()
        if ls.empty:
            st.info("No subject scores available.")
        else:
            top10 = (
                ls.loc[:, ["Subject", "Guardian Score", "Rank"]]
                  .nlargest(10, "Guardian Score")
                  .sort_values("Guardian Score", ascending=True)
            )
            fig_q3 = px.bar(
                top10, x="Guardian Score", y="Subject", orientation="h",
                template=PX_TEMPLATE, color_discrete_sequence=[COLOR_UWE],
                title=f"Top 10 UWE Subjects — Guardian Score ({ly})"
            )
            fig_q3.update_traces(
                text=[f"Rank {int(r)}" if pd.notna(r) else "" for r in top10["Rank"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Guardian Score: %{x:.1f}<br>%{text}<extra></extra>"
            )
            fig_q3.update_layout(xaxis_title="Guardian Score (higher is better)", yaxis_title="", height=520)
            st.plotly_chart(fig_q3, use_container_width=True)

# -----------------------------------------
# TAB 3: Answers (Q1–Q6)
# -----------------------------------------
with tab3:
    st.markdown(f"## Answers to the Strategic Questions — {'Full period' if year_mode=='Full period' else 'Recent 5 years'}")

    # Helper
    def corr_table(df: pd.DataFrame, factor_cols):
        corrs = corr_with_rank(df, factor_cols)
        if corrs.empty:
            return None
        out = corrs.reset_index()
        out.columns = ["Factor", "Correlation with Rank"]
        return out

    # ============== Q1
    st.markdown("### Q1) How has UWE Bristol performed over the period?")
    u = inst_sliced[inst_sliced["Institution"] == UWE_NAME].sort_values("Year")
    if u.empty:
        st.info("No UWE rows in range.")
    else:
        cur_rank = u.loc[u["Year"] == focus_year, "Rank"]
        cur_score = u.loc[u["Year"] == focus_year, "Guardian Score"]
        best_rank = u["Rank"].min()
        slope, _ = lin_trend(u["Year"], u["Rank"])
        trend = "improving (rank ↓)" if slope and slope < 0 else ("worsening (rank ↑)" if slope and slope > 0 else "flat")
        c = st.columns(4)
        c[0].metric("Current Rank", int(cur_rank.iloc[0]) if not cur_rank.empty else "—")
        c[1].metric("Guardian Score", round(float(cur_score.iloc[0]),1) if not cur_score.empty else "—")
        c[2].metric("Best Rank", int(best_rank) if pd.notna(best_rank) else "—")
        c[3].metric("Trend", trend)

        fig_q1 = px.line(u, x="Year", y="Rank", markers=True,
                         template=PX_TEMPLATE, color_discrete_sequence=[COLOR_UWE],
                         title="UWE Rank Over Time")
        fig_q1.update_traces(line=dict(width=4.8), marker=dict(size=10))
        fig_q1.update_yaxes(autorange="reversed", title_text="Rank (lower is better)")
        st.plotly_chart(fig_q1, use_container_width=True)

    # ============== Q2 (split helpful vs harmful, colored)
    st.markdown("### Q2) Which factors improve or worsen UWE’s rank?")
    factor_cols = [c for c in ["Value Added Score","Satisfied with Teaching","Satisfied with Course",
                               "Satisfied with Feedback","Career after 15 months","Student to Staff Ratio"]
                   if c in inst_sliced.columns]
    if factor_cols:
        df_corr = corr_table(inst_sliced, factor_cols)
        if df_corr is not None:
            helpful_df = df_corr[df_corr["Correlation with Rank"] < 0].sort_values("Correlation with Rank")
            harmful_df = df_corr[df_corr["Correlation with Rank"] > 0].sort_values("Correlation with Rank", ascending=False)

            cA, cB = st.columns(2)
            with cA:
                fig_help = px.bar(
                    helpful_df, x="Factor", y="Correlation with Rank",
                    template=PX_TEMPLATE, title="Helpful factors (more → better / lower rank)",
                    color_discrete_sequence=[COLOR_POSITIVE]
                )
                fig_help.update_layout(yaxis_title="Correlation (negative = helpful)", xaxis_title="")
                st.plotly_chart(fig_help, use_container_width=True)
            with cB:
                fig_harm = px.bar(
                    harmful_df, x="Factor", y="Correlation with Rank",
                    template=PX_TEMPLATE, title="Harmful factors (more → worse / higher rank)",
                    color_discrete_sequence=[COLOR_NEGATIVE]
                )
                fig_harm.update_layout(yaxis_title="Correlation (positive = harmful)", xaxis_title="")
                st.plotly_chart(fig_harm, use_container_width=True)
        else:
            st.info("Insufficient data to compute correlations.")
    else:
        st.info("No factor columns available.")

    # ============== Q3 (table remains for quick detail)
    st.markdown("### Q3) How have UWE’s subjects fared over time?")
    if not subj_sliced.empty and "Guardian Score" in subj_sliced.columns:
        ly = int(max(subj_years))
        ls = subj_sliced[subj_sliced["Year"] == ly]
        top10_tbl = ls.nlargest(10, "Guardian Score")[["Subject","Guardian Score","Rank"]].sort_values("Guardian Score", ascending=False)
        st.dataframe(top10_tbl, use_container_width=True)
    else:
        st.info("No subject scores available.")

    # ============== Q4
    st.markdown("### Q4) What’s next for UWE Bristol in the Guardian rankings?")
    if not u.empty and len(u) >= 3:
        s, b0 = lin_trend(u["Year"], u["Rank"])
        next_year = int(max(inst_years)) + 1
        proj = s * next_year + b0 if s is not None else None

        fig_q4 = go.Figure()
        fig_q4.add_trace(go.Scatter(x=u["Year"], y=u["Rank"], mode="lines+markers",
                                    name="UWE Rank", line=dict(color=COLOR_UWE, width=4.8), marker=dict(size=10)))
        if proj is not None:
            fig_q4.add_trace(go.Scatter(x=[u["Year"].min(), next_year],
                                        y=[s*u["Year"].min()+b0, proj],
                                        mode="lines", name="Trend",
                                        line=dict(dash="dash", color=COLOR_HIGHLIGHT, width=2.2)))
            fig_q4.add_trace(go.Scatter(x=[next_year], y=[proj], mode="markers+text", name="Projected",
                                        text=[f"{next_year}: {proj:.1f}"], textposition="bottom center",
                                        marker=dict(color=COLOR_HIGHLIGHT, size=10)))
        fig_q4.update_yaxes(autorange="reversed", title_text="Rank (lower is better)")
        fig_q4.update_layout(template=PX_TEMPLATE, title="Illustrative Rank Projection (direction only)")
        st.plotly_chart(fig_q4, use_container_width=True)
        st.caption("Linear trend for direction only — not a formal forecast.")
    else:
        st.info("Need ≥ 3 years to project.")

    # ============== Q5 (UWE vs Others average — line)
    st.markdown("### Q5) How can UWE perform better in the future?")
    outcome_cols = [c for c in ["Career after 15 months","Satisfied with Feedback","Student to Staff Ratio"]
                    if c in inst_sliced.columns]
    if not outcome_cols:
        st.info("Insufficient data for comparison.")
    else:
        metric = st.selectbox("Choose metric", outcome_cols, index=0, key="ans_metric_ts")

        uwe_ts = (inst_sliced[inst_sliced["Institution"] == UWE_NAME]
                  .sort_values("Year")
                  .groupby("Year", as_index=False)[metric].mean())
        uwe_ts["Series"] = UWE_NAME
        uwe_ts.rename(columns={metric: "Score"}, inplace=True)

        others_ts = (inst_sliced[inst_sliced["Institution"] != UWE_NAME]
                     .sort_values("Year")
                     .groupby("Year", as_index=False)[metric].mean())
        others_ts["Series"] = "Others (avg)"
        others_ts.rename(columns={metric: "Score"}, inplace=True)

        comp_long = pd.concat([uwe_ts, others_ts], ignore_index=True)

        fig_q5 = px.line(
            comp_long, x="Year", y="Score", color="Series", markers=True,
            template=PX_TEMPLATE,
            color_discrete_map={UWE_NAME: COLOR_UWE, "Others (avg)": "#B8BCC2"},
            title=f"{metric} — UWE vs Others (average) over time"
        )
        for tr in fig_q5.data:
            if tr.name == UWE_NAME:
                tr.update(line=dict(width=4.8), marker=dict(size=10), opacity=1.0)
            else:
                tr.update(line=dict(width=2.2), marker=dict(size=7), opacity=0.9)
        st.plotly_chart(fig_q5, use_container_width=True)

        v_uwe = uwe_ts.loc[uwe_ts["Year"] == focus_year, "Score"]
        v_oth = others_ts.loc[others_ts["Year"] == focus_year, "Score"]
        if not v_uwe.empty and not v_oth.empty:
            gap = float(v_uwe.iloc[0]) - float(v_oth.iloc[0])
            direction = "higher (good)" if metric != "Student to Staff Ratio" else "lower (good)"
            st.caption(
                f"At **{focus_year}**, UWE is **{gap:+.1f}** vs others’ average — "
                f"{'higher' if gap > 0 else 'lower'} is {direction} for **{metric}**."
            )

    # ============== Q6
    st.markdown("### Q6) What factors could harm UWE’s league table position?")
    if factor_cols:
        corrs_all = corr_with_rank(inst_sliced, factor_cols)
        if not corrs_all.empty:
            df_harm = corrs_all.sort_values(ascending=False).head(4).reset_index()
            df_harm.columns = ["Factor", "Correlation with Rank"]
            fig_q6 = px.bar(
                df_harm, x="Factor", y="Correlation with Rank",
                template=PX_TEMPLATE,
                title="Most Harmful Correlations (higher → worse rank number)",
                color_discrete_sequence=["#8C2D04"]
            )
            st.plotly_chart(fig_q6, use_container_width=True)

            uh = inst_sliced[inst_sliced["Institution"] == UWE_NAME].sort_values("Year")
            for f in df_harm["Factor"]:
                if f in uh.columns:
                    figh = px.line(
                        uh, x="Year", y=f, markers=True, template=PX_TEMPLATE,
                        title=f"UWE — {f} (higher may harm rank)", color_discrete_sequence=[COLOR_UWE]
                    )
                    figh.update_traces(line=dict(width=4.0))
                    st.plotly_chart(figh, use_container_width=True)
        else:
            st.info("Not enough data to compute harmful factors.")
    else:
        st.info("Required factor columns are not in the data.")
