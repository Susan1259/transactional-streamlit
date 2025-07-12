import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, f_oneway

# ─── 1. APP TITLE ────────────────────────────────────────────────────────────────
st.title("🔍 Transactional Data Analysis")

# ─── 2. LOAD & PREPARE DATA ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Asus\Downloads\bank_transactions_data_2.csv")
    return df

df = load_data()

# ─── 2.1 IDENTIFY NUMERIC AND CATEGORICAL COLUMNS ───────────────────────────────
types = ['int64', 'float64']
numeric_columns = [col for col in df.columns if (df[col].nunique() > 10) & (df[col].dtype in types)]
categoric_columns = [col for col in df.columns if df[col].nunique() < 10]

# ─── 3. SIDEBAR CONTROLS ─────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

# 3.1 Show unique counts
if st.sidebar.checkbox("Show unique value counts", value=False):
    st.sidebar.write(df.nunique())

# 3.2 Filter by TransactionType
tx_types = df.TransactionType.unique().tolist()
selected_types = st.sidebar.multiselect(
    "Filter by TransactionType", tx_types, default=tx_types
)
df = df[df.TransactionType.isin(selected_types)]

# 3.3 Columns to view
view_cols = st.sidebar.multiselect(
    "Columns to view", df.columns.tolist(), default=df.columns[:5].tolist()
)
st.sidebar.markdown("---")

# 3.4 Download filtered data
csv = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "📥 Download Filtered Data", data=csv, file_name="filtered.csv", mime="text/csv"
)

# ─── 4. DATA PREVIEW ─────────────────────────────────────────────────────────────
st.subheader("🔢 Data Preview")
st.dataframe(df[view_cols].head(), height=200)

# ─── 5. DEFINE PLOT FUNCTIONS ────────────────────────────────────────────────────

# Function for Correlation Heatmap using Plotly
def plot_corr_heatmap(df, numeric_columns):
    corr_matrix = df[numeric_columns].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        zmin=-1, zmax=1
    ))
    fig.update_layout(title="Correlation Heatmap (Numeric Features)")
    st.plotly_chart(fig)

# Function for Histograms using Plotly
def plot_numeric_histograms(df, numeric_columns):
    for col in numeric_columns:
        fig = px.histogram(df, x=col, nbins=30, title=f'Histogram of {col}', labels={col: col})
        fig.update_layout(
            width=600, 
            height=400, 
            xaxis_title='',  # Remove x-axis label
            yaxis_title='',  # Remove y-axis label
            xaxis_showgrid=False,  # Optional: Remove grid lines
            yaxis_showgrid=False   # Optional: Remove grid lines
        )
        st.plotly_chart(fig)

# Function for Boxplots using Plotly
def plot_numeric_boxplots(df, numeric_columns):
    for col in numeric_columns:
        fig = px.box(df, y=col, title=f'Boxplot of {col}', labels={col: col})
        fig.update_layout(
            width=600, 
            height=400, 
            xaxis_title='',  # Remove x-axis label
            yaxis_title='',  # Remove y-axis label
            xaxis_showgrid=False,  # Optional: Remove grid lines
            yaxis_showgrid=False   # Optional: Remove grid lines
        )
        st.plotly_chart(fig)

# Function for Transaction Type Countplot using Plotly
def plot_tx_countplot(df):
    fig = px.histogram(df, x='TransactionType', title='Transaction Type Counts', labels={'TransactionType': 'Type'})
    fig.update_layout(width=600, height=400)  # Set consistent size
    st.plotly_chart(fig)

# Function for Scatter Plot (Duration vs Amount) using Plotly
def plot_duration_vs_amount(df):
    fig = px.scatter(df, x='TransactionDuration', y='TransactionAmount', title='Duration vs Transaction Amount', labels={'TransactionDuration': 'Duration (s)', 'TransactionAmount': 'Amount'})
    fig.update_layout(width=600, height=400)  # Set consistent size
    st.plotly_chart(fig)

# Function for Boxplot by Transaction Type for Account Balance using Plotly
def plot_box_account_balance(df):
    fig = px.box(df, x='TransactionType', y='AccountBalance', title='Balance by Transaction Type')
    fig.update_layout(width=600, height=400)  # Set consistent size
    st.plotly_chart(fig)

# Function for Scatter Plot (Age vs Transaction Amount) using Plotly
def plot_age_vs_amount(df):
    fig = px.scatter(df, x='CustomerAge', y='TransactionAmount', color='Channel', trendline='ols', title='Age vs Amount by Channel')
    fig.update_layout(width=600, height=400)  # Set consistent size
    st.plotly_chart(fig)

# Function for Violin Plot (Transaction Type vs Duration) using Plotly
def plot_violin_duration_amount(df):
    fig = px.violin(df, x='TransactionType', y='TransactionDuration', box=True, points="all", title='Duration by Transaction Type', labels={'TransactionDuration': 'Duration (s)', 'TransactionType': 'Type'})
    fig.update_layout(width=600, height=400)  # Set consistent size
    st.plotly_chart(fig)

# Function for Avg Transaction Amount by Occupation using Plotly
def plot_avg_amount_by_occupation(df):
    avg_amt = df.groupby('CustomerOccupation')['TransactionAmount'].mean().sort_values()
    fig = px.bar(avg_amt, x=avg_amt.index, y=avg_amt.values, title='Avg Transaction Amount by Occupation', labels={'x': 'Occupation', 'y': 'Average Amount'})
    fig.update_layout(width=600, height=400)  # Set consistent size
    st.plotly_chart(fig)

# Function for Boxplot for Customer Age by Transaction Type using Plotly
def plot_customer_age_boxplot(df):
    fig = px.box(df, x='TransactionType', y='CustomerAge', title='Customer Age Distribution by Transaction Type')
    fig.update_layout(width=600, height=400)  # Set consistent size
    st.plotly_chart(fig)

# ─── 6. BUTTONS TO TOGGLE NUMERIC AND CATEGORICAL COLUMNS ────────────────────────

# Toggle button for Numeric Columns
if st.button('Show/Hide Numeric Columns', key='numeric_cols', help="Toggle to show/hide numeric columns"):
    if 'show_numeric_columns' not in st.session_state:
        st.session_state.show_numeric_columns = False
    st.session_state.show_numeric_columns = not st.session_state.show_numeric_columns

# Toggle button for Categorical Columns
if st.button('Show/Hide Categorical Columns', key='categoric_cols', help="Toggle to show/hide categorical columns"):
    if 'show_categoric_columns' not in st.session_state:
        st.session_state.show_categoric_columns = False
    st.session_state.show_categoric_columns = not st.session_state.show_categoric_columns

# Show Numeric Columns if button clicked
if st.session_state.get('show_numeric_columns', False):
    st.subheader("🔶 Numeric Columns")
    st.write(numeric_columns)

# Show Categorical Columns if button clicked
if st.session_state.get('show_categoric_columns', False):
    st.subheader("🔶 Categorical Columns")
    st.write(categoric_columns)

# ─── 7. DISPLAY ALL PLOTS ───────────────────────────────────────────────────────
# Heatmap and Univariate Analysis (Numeric Columns)
st.subheader("🔶 Heatmap of Numeric Columns")
plot_corr_heatmap(df, numeric_columns)

st.subheader("📈 Univariate Analysis ")
col1, col2 = st.columns(2)  # Creating two columns for histograms and boxplots side by side
with col1:
    plot_numeric_histograms(df, numeric_columns)

with col2:
    plot_numeric_boxplots(df, numeric_columns)

plot_tx_countplot(df)

# Bivariate Analysis for Numeric Columns
st.subheader("🔀 Bivariate Analysis ")
plot_duration_vs_amount(df)

# Avg Transaction Amount by Occupation
plot_avg_amount_by_occupation(df)

# Transaction Type Boxplot by Account Balance
plot_box_account_balance(df)

# Customer Age Boxplot by Transaction Type
plot_customer_age_boxplot(df)

# Duration by Transaction Type
plot_violin_duration_amount(df)

# ─── 10. AGE GROUPS AND TRANSACTION CHANNELS ────────────────────────────────────
# Create age groups
df['age_group'] = pd.cut(
    df['CustomerAge'],
    bins=[0, 25, 40, 60, 100],
    labels=['0–25', '26–40', '41–60', '61+']
)

# Aggregate counts per age_group & channel
age_channel = (
    df
    .groupby(['age_group', 'Channel'])
    .size()
    .reset_index(name='count')
)

# Plot a grouped bar chart
fig = px.bar(
    age_channel,
    x='age_group',
    y='count',
    color='Channel',
    barmode='group',
    title='Transaction Channel by Age Group',
    labels={'age_group': 'Age Group', 'count': 'Number of Transactions'}
)

fig.update_layout(xaxis_title='Age Group', yaxis_title='Transactions')
st.plotly_chart(fig)

# ─── 9. CHI-SQUARE AND ANOVA TESTS ───────────────────────────────────────────────

st.subheader("📊 Chi-Square Tests")

# Chi-Square test
columns = df[categoric_columns].columns
for i in range(len(columns)):
    for k in range(i + 1, len(columns)):
        table = pd.crosstab(df[columns[i]], df[columns[k]])
        chi2, p, dof, expected = chi2_contingency(table)
        st.write(f"{columns[i]} vs {columns[k]} → chi2 = {chi2:.2f}, p = {p:.4f}")

# ANOVA test
st.subheader("📊 Anova Tests")
cat_cols = df[categoric_columns].columns
num_cols = df[numeric_columns].columns

for cat in cat_cols:
    for num in num_cols:
        try:
            temp = df[[cat, num]].dropna()
            groups = [group[num].values for name, group in temp.groupby(cat)]
            if len(groups) > 1:
                f_stat, p_val = f_oneway(*groups)
                st.write(f"{num} by {cat} → F = {f_stat:.2f}, p = {p_val:.4f}")
        except Exception as e:
            st.write(f"Skipped {cat} vs {num}: {e}")
