import os
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# Theme & Palette
# =========================
st.set_page_config(page_title="UWE Bristol Dashboard", layout="wide")
PX_TEMPLATE = "plotly_white"

UWE_NAME = "UWE Bristol"
COLOR_UWE = "#0072B2"             # UWE highlight (bold blue)
COLOR_COMPETITORS = [             # distinct, professional competitor colors
    "#6C757D",  # slate
    "#808000",  # olive
    "#800000",  # burgundy
    "#6A5ACD",  # muted purple
    "#A0522D",  # sienna
    "#9467BD",  # violet
    "#8C564B"   # brown
]
COLOR_HIGHLIGHT = "#E69F00"       # callouts/annotations
COLOR_POSITIVE = "#009E73"        # positive indicators
TEXT_DARK = "#111827"

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
    if mode == "Full period":
        dom = years
    else:
        dom = years[-RECENT_WINDOW_SIZE:]
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

def yoy(df: pd.DataFrame, cols, group="Institution"):
    out = df.sort_values([group,"Year"]).copy()
    for c in cols:
        if c in out.columns:
            out[f"Δ {c}"] = out.groupby(group)[c].diff()
    return out

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
    st.markdown("### For understanding")

    c1, c2, c3 = st.columns(3)

    # Satisfaction donut (UWE, focus year)
    uwe_year = inst_sliced[(inst_sliced["Institution"] == UWE_NAME) & (inst_sliced["Year"] == focus_year)]
    if not uwe_year.empty and {"Satisfied with Course","Satisfied with Teaching","Satisfied with Feedback"}.issubset(uwe_year.columns):
        sat = pd.DataFrame({
            "Metric": ["Course", "Teaching", "Feedback"],
            "Value": [
                float(uwe_year["Satisfied with Course"].iloc[0]),
                float(uwe_year["Satisfied with Teaching"].iloc[0]),
                float(uwe_year["Satisfied with Feedback"].iloc[0]),
            ]
        })
        fig_pie = px.pie(
            sat, values="Value", names="Metric", hole=0.45,
            title=f"Satisfaction • UWE ({focus_year})", template=PX_TEMPLATE,
            color_discrete_sequence=[COLOR_UWE, "#66A9CF", "#A6CEE3"]
        )
        c1.plotly_chart(fig_pie, use_container_width=True)
    else:
        c1.info("Satisfaction metrics not available for focus year.")

    # Dual-axis: Rank (left, reversed) vs Guardian Score (right) — UWE only
    uwe_series = inst_sliced[inst_sliced["Institution"] == UWE_NAME].sort_values("Year")
    if not uwe_series.empty:
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(
            x=uwe_series["Year"], y=uwe_series["Rank"], name="Rank (lower is better)",
            mode="lines+markers", line=dict(color=COLOR_UWE, width=4.8), marker=dict(size=10)
        ))
        fig_dual.add_trace(go.Scatter(
            x=uwe_series["Year"], y=uwe_series["Guardian Score"], name="Guardian Score",
            mode="lines+markers", yaxis="y2",
            line=dict(color="#333333", width=2.2, dash="dot"), marker=dict(size=7)
        ))
        fig_dual.update_layout(
            template=PX_TEMPLATE, title="Rank vs Guardian Score (UWE)",
            yaxis=dict(title="Rank", autorange="reversed"),
            yaxis2=dict(title="Guardian Score", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        c2.plotly_chart(fig_dual, use_container_width=True)
    else:
        c2.info("No UWE rows in selected time range.")

    # Satisfaction bars (UWE)
    sat_cols = [c for c in ["Satisfied with Feedback","Satisfied with Course"] if c in inst_sliced.columns]
    if sat_cols:
        subset = inst_sliced[inst_sliced["Institution"] == UWE_NAME].sort_values("Year")
        if not subset.empty:
            fig_sat = px.bar(
                subset, x="Year", y=sat_cols, barmode="group",
                title="Satisfaction (UWE)", template=PX_TEMPLATE,
                color_discrete_map={
                    "Satisfied with Feedback": "#66A9CF",
                    "Satisfied with Course": "#A6CEE3"
                }
            )
            c3.plotly_chart(fig_sat, use_container_width=True)
        else:
            c3.info("No UWE rows to display.")
    else:
        c3.info("No satisfaction columns in data.")

    st.markdown("#### Snapshot (UWE)")
    cols_to_show = [c for c in ["Year","Rank","Guardian Score","Satisfied with Course","Satisfied with Teaching","Satisfied with Feedback"] if c in inst_sliced.columns]
    uwetable = inst_sliced[inst_sliced["Institution"] == UWE_NAME][cols_to_show].sort_values("Year", ascending=False)
    st.dataframe(uwetable, use_container_width=True)

    st.divider()

    # Comparison: Rank Over Time (earliest → latest)
    st.markdown("### Comparison — Rank Over Time")
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

# -----------------------------------------
# TAB 2: Subjects
# -----------------------------------------
with tab2:
    st.markdown("### Subject Performance (UWE)")
    if subj_sliced.empty:
        st.info("No UWE subject data in this time range.")
    else:
        c1, c2 = st.columns([1.2, 1.8])
        year_subject = c1.slider("Year (Top‑10 view)", min_value=int(min(subj_years)),
                                 max_value=int(max(subj_years)), value=int(max(subj_years)), step=1)
        latest_slice = subj_sliced[subj_sliced["Year"] == year_subject].copy()
        if not latest_slice.empty and "Guardian Score" in latest_slice.columns:
            top10 = latest_slice.nlargest(10, "Guardian Score").sort_values("Guardian Score")
            fig_top = px.bar(
                top10, x="Guardian Score", y="Subject", orientation="h",
                title=f"Top 10 Subjects by Guardian Score ({year_subject})",
                template=PX_TEMPLATE, color_discrete_sequence=[COLOR_UWE]
            )
            c1.plotly_chart(fig_top, use_container_width=True)
        else:
            c1.info("No Guardian Score for selected year.")

        subj_choice = c2.selectbox("Choose Subject", sorted(subj_sliced["Subject"].unique()))
        sdata = subj_sliced[subj_sliced["Subject"] == subj_choice].sort_values("Year")
        if sdata.empty:
            c2.info("No data for selected subject.")
        else:
            fig_s = px.line(
                sdata, x="Year", y="Rank", markers=True,
                title=f"{subj_choice} — UWE Rank Over Time",
                template=PX_TEMPLATE, color_discrete_sequence=[COLOR_UWE]
            )
            fig_s.update_traces(line=dict(width=4.8), marker=dict(size=10))
            fig_s.update_yaxes(autorange="reversed")
            c2.plotly_chart(fig_s, use_container_width=True)

# -----------------------------------------
# TAB 3: Answers (Q1–Q6)
# -----------------------------------------
with tab3:
    st.markdown(f"## Answers to the Strategic Questions — {'Full period' if year_mode=='Full period' else 'Recent 5 years'}")

    # Helper for correlation blocks
    def corr_block(df: pd.DataFrame, factor_cols):
        corrs = corr_with_rank(df, factor_cols)
        if corrs.empty:
            return None, None, None
        df_corr = corrs.reset_index()
        df_corr.columns = ["Factor", "Correlation with Rank"]
        helpful = df_corr.sort_values("Correlation with Rank", ascending=True).head(3)
        harmful = df_corr.sort_values("Correlation with Rank", ascending=False).head(3)
        return df_corr, helpful, harmful

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
        fig_q1.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_q1, use_container_width=True)

    # ============== Q2
    st.markdown("### Q2) What factors explain UWE’s performance?")
    factor_cols = [c for c in ["Value Added Score","Satisfied with Teaching","Satisfied with Course",
                               "Satisfied with Feedback","Career after 15 months","Student to Staff Ratio"]
                   if c in inst_sliced.columns]
    if factor_cols:
        df_corr, helpful, harmful = corr_block(inst_sliced, factor_cols)
        if df_corr is not None:
            fig_q2 = px.bar(
                df_corr, x="Factor", y="Correlation with Rank",
                template=PX_TEMPLATE,
                title="Correlation of Factors with Rank (lower rank is better)",
                color_discrete_sequence=[COLOR_POSITIVE]
            )
            st.plotly_chart(fig_q2, use_container_width=True)
            cA, cB = st.columns(2)
            with cA:
                st.write("**Most Helpful (more → better/lower rank)**")
                st.dataframe(helpful, use_container_width=True)
            with cB:
                st.write("**Most Harmful (more → worse/higher rank)**")
                st.dataframe(harmful, use_container_width=True)
        else:
            st.info("Insufficient data to compute correlations.")
    else:
        st.info("No factor columns available.")

    # ============== Q3
    st.markdown("### Q3) How have UWE’s subjects fared over time?")
    if not subj_sliced.empty and "Guardian Score" in subj_sliced.columns:
        ly = int(max(subj_years))
        ls = subj_sliced[subj_sliced["Year"] == ly]
        top10 = ls.nlargest(10, "Guardian Score")[["Subject","Guardian Score","Rank"]].sort_values("Guardian Score", ascending=False)
        st.dataframe(top10, use_container_width=True)
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
        fig_q4.update_yaxes(autorange="reversed")
        fig_q4.update_layout(template=PX_TEMPLATE, title="Illustrative Rank Projection (direction only)")
        st.plotly_chart(fig_q4, use_container_width=True)
        st.caption("Linear trend for direction only — not a formal forecast.")
    else:
        st.info("Need ≥ 3 years to project.")

    # ============== Q5
    st.markdown("### Q5) How can UWE perform better in the future?")
    outcome_cols = [c for c in ["Career after 15 months","Satisfied with Feedback","Student to Staff Ratio"] if c in inst_sliced.columns]
    comp_year = inst_sliced[inst_sliced["Year"] == focus_year]
    competitors = [i for i in sorted(comp_year["Institution"].unique().tolist()) if i != UWE_NAME]
    if competitors and not comp_year.empty and outcome_cols:
        comp_choice = st.selectbox("Choose competitor", competitors, key="ans_comp")
        uwe_row = comp_year[comp_year["Institution"] == UWE_NAME]
        cmp_row = comp_year[comp_year["Institution"] == comp_choice]
        if not uwe_row.empty and not cmp_row.empty:
            gap_df = pd.DataFrame({
                "Metric": outcome_cols,
                UWE_NAME: [uwe_row[m].iloc[0] for m in outcome_cols],
                comp_choice: [cmp_row[m].iloc[0] for m in outcome_cols]
            })
            long_gap = gap_df.melt(id_vars="Metric", var_name="Institution", value_name="Score")
            color_map_bar = {UWE_NAME: COLOR_UWE, comp_choice: fixed_color_map([UWE_NAME, comp_choice])[comp_choice]}
            fig_q5 = px.bar(
                long_gap, x="Metric", y="Score", color="Institution", barmode="group",
                title=f"Gaps vs {comp_choice} — improvement levers ({focus_year})",
                template=PX_TEMPLATE, color_discrete_map=color_map_bar
            )
            st.plotly_chart(fig_q5, use_container_width=True)

            # quick “focus” suggestion (where UWE trails)
            eval_df = gap_df.copy()
            # For SSR lower is better → flip sign so negative means UWE behind for all metrics
            eval_df["UWE minus Competitor"] = [
                (eval_df.loc[i,UWE_NAME] - eval_df.loc[i,comp_choice]) if m != "Student to Staff Ratio"
                else (eval_df.loc[i,comp_choice] - eval_df.loc[i,UWE_NAME])
                for i, m in enumerate(eval_df["Metric"])
            ]
            focus = eval_df.sort_values("UWE minus Competitor").head(2)["Metric"].tolist()
            st.success("**Recommended focus:** " + (", ".join(focus) if focus else "—"))
        else:
            st.info("Missing data for UWE or selected competitor.")
    else:
        st.info("Insufficient data for gap analysis.")

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
            # UWE trend for each harmful factor
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
