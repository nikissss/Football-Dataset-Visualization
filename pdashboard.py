import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/nikki/Downloads/players_PP.csv")
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['Age'] = datetime.now().year - df['date_of_birth'].dt.year
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
positions = st.sidebar.multiselect("Select Position", options=df['position'].unique(), default=df['position'].unique())
# clubs = st.sidebar.multiselect("Select Club", options=df['current_club_name'].unique(), default=df['current_club_name'].unique())
countries = st.sidebar.multiselect("Select Nationality", options=df['country_of_citizenship'].dropna().unique(), default=df['country_of_citizenship'].dropna().unique())

# Filtered Data
filtered_df = df[
    (df['position'].isin(positions)) &
    # (df['current_club_name'].isin(clubs)) &
    (df['country_of_citizenship'].isin(countries))
]

# Title
st.title("⚽ Football Players Dashboard")
# st.markdown("Interactive dashboard showing player data with Plotly and Streamlit.")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Players", len(filtered_df))
col2.metric("Avg Market Value (€)", f"{filtered_df['market_value_in_eur'].mean():,.0f}")
col3.metric("Avg Age", f"{filtered_df['Age'].mean():.1f}")

# Market Value Distribution
left, right = st.columns(2)
with left:
    st.subheader("Market Value Distribution")
    fig1 = px.histogram(filtered_df, x="market_value_in_eur", nbins=50,)
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("Market Value by Position")
    fig2 = px.box(filtered_df, x="position", y="market_value_in_eur", points="all", )
    st.plotly_chart(fig2, use_container_width=True)

# Age vs Market Value
st.subheader("Age vs Market Value")
fig3 = px.scatter(filtered_df, x="Age", y="market_value_in_eur", color="position", )
st.plotly_chart(fig3, use_container_width=True)

# Foot Preference and Grouped Bar
left2, right2 = st.columns(2)
with left2:
    st.subheader("Foot Preference Distribution")
    fig4 = px.pie(filtered_df, names="foot", )
    st.plotly_chart(fig4, use_container_width=True)

with right2:
    st.subheader("Avg Market Value by Position and Foot")
    grouped = filtered_df.groupby(['position', 'foot'])['market_value_in_eur'].mean().reset_index()
    fig6 = px.bar(grouped, x="position", y="market_value_in_eur", color="foot", barmode="group", )
    st.plotly_chart(fig6, use_container_width=True)

# Age Distribution
st.subheader("Age Distribution of Players")
fig7 = px.histogram(filtered_df, x="Age", nbins=40, )
st.plotly_chart(fig7, use_container_width=True)

# Most Valuable Defender
st.subheader("Most Valuable Defenders")
defenders = df[df['position'] == "Defender"].sort_values(by="market_value_in_eur", ascending=False).head(10)
st.table(defenders[['name', 'position', 'market_value_in_eur', 'current_club_name']])

# Most Valuable Goalkeepers
st.subheader("Most Valuable Goalkeepers")
keepers = df[df['position'] == "Goalkeeper"].sort_values(by="market_value_in_eur", ascending=False).head(10)
st.table(keepers[['name', 'position', 'market_value_in_eur', 'current_club_name']])

# Most Valuable XI (Based on Sub-Position with 2 CBs)
st.subheader("Most Valuable XI")
xi_positions = [
    ('Goalkeeper', 1),
    ('Right-Back', 1),
    ('Left-Back', 1),
    ('Centre-Back', 2),
    ('Defensive Midfield', 1),
    ('Central Midfield', 1),
    ('Attacking Midfield', 1),
    ('Left Winger', 1),
    ('Right Winger', 1),
    ('Centre-Forward', 1),
]

xi_players = pd.DataFrame()
for sub_pos, count in xi_positions:
    selected = df[df['sub_position'] == sub_pos].sort_values(by='market_value_in_eur', ascending=False).head(count)
    xi_players = pd.concat([xi_players, selected])

fig8 = px.bar(xi_players, x="name", y="market_value_in_eur", color="sub_position", title="Top 11 Most Valuable Players by Sub-Position")
st.plotly_chart(fig8, use_container_width=True)

# Most Valuable Clubs
st.subheader("Top 10 Most Valuable Clubs")
club_value = df.groupby('current_club_name')['market_value_in_eur'].sum().reset_index().sort_values(by='market_value_in_eur', ascending=False).head(10)
st.table(club_value)

# Table Preview
st.subheader("Top Players by Market Value")
st.dataframe(filtered_df[['name', 'position', 'Age', 'market_value_in_eur', 'current_club_name']].sort_values(by="market_value_in_eur", ascending=False).head(20))
