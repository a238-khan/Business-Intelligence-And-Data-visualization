import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------
# Load the Excel data
# -------------------------
@st.cache_data
def load_guardian_data():
    file_path = "../Guardian_Cleaned_Final.xlsx"  # Path relative to visualization folder
    sheets = pd.read_excel(file_path, sheet_name=None)
    return sheets

data = load_guardian_data()
inst_data = data["Institution Level Data"]
subj_data = data["Subject Level Data"]

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="UWE Bristol Guardian Dashboard", layout="wide")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ Performance Over Time",
    "2️⃣ Factors Explaining Performance",
    "3️⃣ Subject Performance",
    "4️⃣ Momentum (What's Next)",
    "5️⃣ Improvement Opportunities",
    "6️⃣ Risk Monitoring"
])

# -------------------------
# TAB 1: Performance Over Time
# -------------------------
with tab1:
    st.subheader("How has UWE Bristol performed over the period?")
    competitors = st.multiselect(
        "Select Competitors",
        sorted(inst_data['Institution'].unique()),
        default=["UWE Bristol", "University of Bristol", "Bath Spa University", "Cardiff Metropolitan University"]
    )

    df_filtered = inst_data[inst_data['Institution'].isin(competitors)]
    fig = px.line(
        df_filtered,
        x="Year",
        y="Rank",
        color="Institution",
        markers=True,
        title="Rank Over Time"
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    fig_score = px.line(
        df_filtered,
        x="Year",
        y="Guardian Score",
        color="Institution",
        markers=True,
        title="Guardian Score Over Time"
    )
    st.plotly_chart(fig_score, use_container_width=True)

# -------------------------
# TAB 2: Factors Explaining Performance
# -------------------------
with tab2:
    st.subheader("What factors explain UWE’s performance?")
    latest_year = inst_data['Year'].max()
    st.write(f"Latest Year: {latest_year}")

    competitors = st.multiselect(
        "Select Competitors",
        sorted(inst_data['Institution'].unique()),
        default=["UWE Bristol", "University of Bristol", "Bath Spa University"]
    )

    metrics = ['Value Added Score', 'Satisfied with Teaching', 'Satisfied with Course', 'Career after 15 months']
    df_latest = inst_data[(inst_data['Year'] == latest_year) & (inst_data['Institution'].isin(competitors))]

    fig = px.bar(
        df_latest,
        x="Institution",
        y=metrics,
        barmode="group",
        title="Key Performance Factors (Latest Year)"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# TAB 3: Subject Performance
# -------------------------
with tab3:
    st.subheader("How have UWE’s subjects fared over the period?")
    uwe_subjects = subj_data[subj_data['Institution'] == "UWE Bristol"]

    subject_choice = st.selectbox("Select Subject", sorted(uwe_subjects['Subject'].unique()))
    subject_data = uwe_subjects[uwe_subjects['Subject'] == subject_choice]

    fig = px.line(
        subject_data,
        x="Year",
        y="Rank",
        markers=True,
        title=f"{subject_choice} - Rank Over Time"
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    # Table of top 10 subjects latest year
    top_subjects = uwe_subjects[uwe_subjects['Year'] == uwe_subjects['Year'].max()].nsmallest(10, 'Rank')
    st.write("Top 10 Subjects (Latest Year)")
    st.dataframe(top_subjects[['Subject', 'Rank', 'Guardian Score']])

# -------------------------
# TAB 4: Momentum (What's Next)
# -------------------------
with tab4:
    st.subheader("What’s next for UWE Bristol in the Guardian rankings?")
    uwe = inst_data[inst_data['Institution'] == 'UWE Bristol'].sort_values('Year')
    uwe['Guardian_score_change'] = uwe['Guardian Score'].diff()
    uwe['Teaching_change'] = uwe['Satisfied with Teaching'].diff()
    uwe['Career_change'] = uwe['Career after 15 months'].diff()

    fig = px.line(
        uwe,
        x="Year",
        y=['Guardian_score_change', 'Teaching_change', 'Career_change'],
        markers=True,
        title="Year-on-Year Change in Key KPIs"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# TAB 5: Improvement Opportunities
# -------------------------
with tab5:
    st.subheader("How can UWE Bristol perform better in the future?")
    latest_year = inst_data['Year'].max()
    uwe_latest = inst_data[(inst_data['Institution'] == 'UWE Bristol') & (inst_data['Year'] == latest_year)]

    competitor = st.selectbox(
        "Select Competitor",
        [u for u in inst_data['Institution'].unique() if u != 'UWE Bristol']
    )
    comp_latest = inst_data[(inst_data['Institution'] == competitor) & (inst_data['Year'] == latest_year)]

    metrics_gap = ['Career after 15 months', 'Student to Staff Ratio', 'Satisfied with Feedback']
    gap_df = pd.DataFrame({
        'Metric': metrics_gap,
        'UWE Bristol': uwe_latest[metrics_gap].values[0],
        competitor: comp_latest[metrics_gap].values[0]
    })

    fig = px.bar(
        gap_df.melt(id_vars='Metric', var_name='Institution', value_name='Score'),
        x="Metric",
        y="Score",
        color="Institution",
        barmode="group",
        title=f"Gap Analysis: UWE vs {competitor}"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# TAB 6: Risk Monitoring
# -------------------------
with tab6:
    st.subheader("What factors could harm UWE’s league table position?")
    competitors = st.multiselect(
        "Select Competitors",
        sorted(inst_data['Institution'].unique()),
        default=["UWE Bristol", "University of Bristol", "Bath Spa University"]
    )

    metrics = ['Value Added Score', 'Satisfied with Teaching']
    for metric in metrics:
        fig = px.line(
            inst_data[inst_data['Institution'].isin(competitors)],
            x="Year",
            y=metric,
            color="Institution",
            markers=True,
            title=f"{metric} Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
