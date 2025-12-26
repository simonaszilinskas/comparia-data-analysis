import streamlit as st
import pandas as pd
from datasets import load_dataset

# Page config
st.set_page_config(
    page_title="compar:IA - Visualizations",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .nav-container {
        display: flex;
        gap: 10px;
        margin-bottom: 2rem;
        justify-content: center;
    }
    .nav-button {
        padding: 12px 32px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 8px;
        text-decoration: none;
        transition: all 0.2s;
        border: 2px solid #e0e0e0;
        display: inline-block;
    }
    .nav-button-selected {
        background-color: #0068c9;
        color: white !important;
        border-color: #0068c9;
    }
    .nav-button-unselected {
        background-color: white;
        color: #0068c9 !important;
        border-color: #0068c9;
    }
    .nav-button-unselected:hover {
        background-color: #f0f8ff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_conversations(sample_size=10000):
    """Load conversations dataset"""
    ds = load_dataset('ministere-culture/comparia-conversations', split=f'train[:{sample_size}]')
    df = ds.to_pandas()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def main():
    # Custom navigation buttons at the top
    st.markdown("""
    <div class="nav-container">
        <a href="/" target="_self" class="nav-button nav-button-unselected">🔍 Search</a>
        <a href="/Visualizations" target="_self" class="nav-button nav-button-selected">📊 Visualizations</a>
    </div>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="main-title">📊 compar:IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dataset visualizations and statistics</div>', unsafe_allow_html=True)

    # Load data
    sample_size = 10000  # Fixed sample size
    with st.spinner("Loading data..."):
        df = load_conversations(sample_size)

    # Basic statistics
    st.markdown("### 📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Conversations", f"{len(df):,}")

    with col2:
        unique_models = len(set(df['model_a_name'].unique()) | set(df['model_b_name'].unique()))
        st.metric("Unique Models", unique_models)

    with col3:
        # Count unique categories
        all_categories = []
        for cat_list in df['categories']:
            if cat_list is not None and hasattr(cat_list, '__iter__') and len(cat_list) > 0:
                cats = list(cat_list) if not isinstance(cat_list, list) else cat_list
                all_categories.extend(cats)
        st.metric("Categories", len(set(all_categories)))

    with col4:
        avg_turns = df['conv_turns'].mean() if 'conv_turns' in df.columns else 0
        st.metric("Avg Conversation Turns", f"{avg_turns:.1f}")

    st.markdown("---")

    # Top models
    st.markdown("### 🤖 Most Frequent Models")
    model_counts = pd.concat([df['model_a_name'], df['model_b_name']]).value_counts().head(10)

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(model_counts)

    with col2:
        st.dataframe(
            pd.DataFrame({'Model': model_counts.index, 'Count': model_counts.values}),
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")

    # Categories distribution
    st.markdown("### 🏷️ Top Categories")
    if all_categories:
        cat_counts = pd.Series(all_categories).value_counts().head(15)
        st.bar_chart(cat_counts)

if __name__ == "__main__":
    main()
