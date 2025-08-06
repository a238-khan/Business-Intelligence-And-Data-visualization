import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="UWE Bristol Guardian Dashboard", layout="wide")

# -------------------------
# File Path
# -------------------------
FILE_PATH = "visualization/Guardian_Cleaned_Final.xlsx"

# -------------------------
# Cached Loader
# -------------------------
@st.cache_data
def load_sheet(sheet_name):
    return pd.read_excel(FILE_PATH, sheet_name=sheet_name)

# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs([
    "🏛 Institution Performance",
    "📚 Subject Performance"
])

# ============================================================
# TAB 1 – Institution Performance
# ============================================================
with tab1:
    st.markdown("## UWE Bristol Performance Overview")

    with st.spinner("Loading data..."):
        inst_data = load_sheet("Institution Level Data")

    # --- KPI Cards ---
    latest_year = inst_data['Year'].max()
    uwe_latest = inst_data[(inst_data['Institution'] == 'UWE Bristol') & (inst_data['Year'] == latest_year)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rank", int(uwe_latest['Rank'].values[0]))
    col2.metric("Guardian Score", round(uwe_latest['Guardian Score'].values[0], 1))
    col3.metric("Top Subject Rank", int(uwe_latest['Rank'].min()))

    # --- Top Row: Pie | Line | Bar ---
    col1, col2, col3 = st.columns(3)

    # Pie Chart
    pie_data = {
        'Guardian Score': uwe_latest['Guardian Score'].values[0],
        'Rank': uwe_latest['Rank'].values[0],
        'Other': 100 - (uwe_latest['Guardian Score'].values[0] + uwe_latest['Rank'].values[0])
    }
    fig_pie = px.pie(values=pie_data.values(), names=pie_data.keys(), hole=0.4, title="Latest Year KPI Breakdown")
    col1.plotly_chart(fig_pie, use_container_width=True)

    # Line Chart: Rank & Score over time
    competitors_inst = st.multiselect(
        "Select Competitors",
        sorted(inst_data['Institution'].unique()),
        default=["UWE Bristol", "Bristol", "Bath Spa", "Cardiff Met"],
        key="competitors_inst"
    )
    df_filtered = inst_data[inst_data['Institution'].isin(competitors_inst)]
    fig_line = px.line(df_filtered, x="Year", y="Rank", color="Institution", markers=True, title="Rank Over Time")
    fig_line.update_yaxes(autorange="reversed")
    col2.plotly_chart(fig_line, use_container_width=True)

    # Bar Chart: Satisfaction metrics
    fig_bar = px.bar(uwe_latest, x="Year", y=["Satisfied with Feedback", "Satisfied with Course"],
                     barmode="group", title="Satisfaction Metrics (Latest Year)")
    col3.plotly_chart(fig_bar, use_container_width=True)

    # --- Exploration: Value Added etc. ---
    st.markdown("### Performance Factors vs Competitors")
    competitors_factors = st.multiselect(
        "Select Competitors for Factors",
        sorted(inst_data['Institution'].unique()),
        default=["UWE Bristol", "Bristol", "Bath Spa"],
        key="competitors_factors"
    )
    metrics = ['Value Added Score', 'Satisfied with Teaching', 'Satisfied with Course', 'Career after 15 months']
    df_latest = inst_data[(inst_data['Year'] == latest_year) & (inst_data['Institution'].isin(competitors_factors))]
    fig_factors = px.bar(df_latest, x="Institution", y=metrics, barmode="group", title="Key Performance Factors")
    st.plotly_chart(fig_factors, use_container_width=True)

    # --- Resolution: Gap Analysis ---
    st.markdown("### Gap Analysis vs Competitor")
    competitor_choice = st.selectbox(
        "Select Competitor for Gap Analysis",
        [u for u in inst_data['Institution'].unique() if u != 'UWE Bristol'],
        key="competitor_gap"
    )
    comp_latest = inst_data[(inst_data['Institution'] == competitor_choice) & (inst_data['Year'] == latest_year)]
    metrics_gap = ['Career after 15 months', 'Student to Staff Ratio', 'Satisfied with Feedback']
    gap_df = pd.DataFrame({
        'Metric': metrics_gap,
        'UWE Bristol': uwe_latest[metrics_gap].values[0],
        competitor_choice: comp_latest[metrics_gap].values[0]
    })
    fig_gap = px.bar(gap_df.melt(id_vars='Metric', var_name='Institution', value_name='Score'),
                     x="Metric", y="Score", color="Institution", barmode="group",
                     title=f"Gap Analysis: UWE vs {competitor_choice}")
    st.plotly_chart(fig_gap, use_container_width=True)

    # --- Table ---
    st.markdown("### Yearly Institution Data")
    st.dataframe(inst_data[inst_data['Institution'] == 'UWE Bristol']
                 [['Year', 'Rank', 'Guardian Score', 'Satisfied with Course',
                   'Satisfied with Teaching', 'Satisfied with Feedback']])

# ============================================================
# TAB 2 – Subject Performance
# ============================================================
with tab2:
    st.markdown("## UWE Bristol Subject Performance")

    with st.spinner("Loading subject data..."):
        subj_data = load_sheet("Subject Level Data")

    uwe_subjects = subj_data[subj_data['Institution'] == "UWE Bristol"]
    subject_choice = st.selectbox("Select Subject", sorted(uwe_subjects['Subject'].unique()), key="subject_select")
    subject_data = uwe_subjects[uwe_subjects['Subject'] == subject_choice]

    # Line Chart: Subject Rank over time
    fig_subject_line = px.line(subject_data, x="Year", y="Rank", markers=True,
                               title=f"{subject_choice} - Rank Over Time")
    fig_subject_line.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_subject_line, use_container_width=True)

    # Top 10 subjects latest year
    st.markdown("### Top 10 Subjects (Latest Year)")
    top_subjects = uwe_subjects[uwe_subjects['Year'] == uwe_subjects['Year'].max()].nsmallest(10, 'Rank')
    st.dataframe(top_subjects[['Subject', 'Rank', 'Guardian Score']])
