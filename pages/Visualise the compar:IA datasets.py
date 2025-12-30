import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datasets import load_dataset

# Page config
st.set_page_config(
    page_title="Visualise the compar:IA datasets",
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
    /* Hide the main app page from sidebar */
    [data-testid="stSidebarNav"] li:first-child {
        display: none;
    }
    /* Align button text to the left */
    [data-testid="stSidebar"] button[kind="secondary"] p {
        text-align: left !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_conversations(sample_size=10000):
    """Load conversations dataset with memory optimizations"""
    import os
    import sys

    # Get HuggingFace token from environment or Streamlit secrets
    token = os.environ.get("HF_TOKEN")
    if not token and hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets:
        token = st.secrets["HF_TOKEN"]

    # Suppress progress bars to avoid BrokenPipeError
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    try:
        # Only load columns needed for visualizations
        columns_needed = [
            'conversation_pair_id', 'model_a_name', 'model_b_name',
            'timestamp', 'categories', 'conv_turns'
        ]

        if sample_size is None:
            # Load full dataset with only needed columns
            ds = load_dataset('ministere-culture/comparia-conversations', split='train', token=token)
        else:
            # Load sample
            ds = load_dataset('ministere-culture/comparia-conversations', split=f'train[:{sample_size}]', token=token)

        # Convert to pandas and optimize memory
        df = ds.to_pandas()

        # Keep only needed columns if they exist
        available_cols = [col for col in columns_needed if col in df.columns]
        if available_cols:
            df = df[available_cols]

        # Optimize data types to reduce memory usage
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Convert model names to category type (saves memory for repeated values)
        if 'model_a_name' in df.columns:
            df['model_a_name'] = df['model_a_name'].astype('category')
        if 'model_b_name' in df.columns:
            df['model_b_name'] = df['model_b_name'].astype('category')

        return df
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr

@st.cache_data
def load_votes():
    """Load votes and reactions datasets with user preferences"""
    import os
    import sys

    # Get HuggingFace token from environment or Streamlit secrets
    token = os.environ.get("HF_TOKEN")
    if not token and hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets:
        token = st.secrets["HF_TOKEN"]

    # Suppress progress bars to avoid BrokenPipeError
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    try:
        # Load votes dataset
        ds_votes = load_dataset('ministere-culture/comparia-votes', split='train', token=token)
        df_votes = ds_votes.to_pandas()
        df_votes['timestamp'] = pd.to_datetime(df_votes['timestamp'])
        df_votes['source'] = 'votes'
        votes_cols = df_votes[['conversation_pair_id', 'chosen_model_name', 'both_equal', 'source']]

        # Load reactions dataset
        ds_reactions = load_dataset('ministere-culture/comparia-reactions', split='train', token=token)
        df_reactions = ds_reactions.to_pandas()
        df_reactions['timestamp'] = pd.to_datetime(df_reactions['timestamp'])
        df_reactions['source'] = 'reactions'

        # Check what columns reactions dataset has and adapt
        available_cols = ['conversation_pair_id', 'source']

        if 'chosen_model_name' in df_reactions.columns:
            available_cols.append('chosen_model_name')
        elif 'winner' in df_reactions.columns:
            df_reactions['chosen_model_name'] = df_reactions['winner']
            available_cols.append('chosen_model_name')

        if 'both_equal' in df_reactions.columns:
            available_cols.append('both_equal')
        elif 'tie' in df_reactions.columns:
            df_reactions['both_equal'] = df_reactions['tie']
            available_cols.append('both_equal')

        reactions_cols = df_reactions[available_cols]

        if 'chosen_model_name' not in reactions_cols.columns:
            reactions_cols['chosen_model_name'] = None
        if 'both_equal' not in reactions_cols.columns:
            reactions_cols['both_equal'] = False

        # Combine both datasets
        df_combined = pd.concat([votes_cols, reactions_cols], ignore_index=True)

        return df_combined
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr

def main():
    # Settings in sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Initialize sample size in session state if not exists
        if 'sample_size_index' not in st.session_state:
            st.session_state.sample_size_index = 2  # Default to 10,000

        # Sample size selector
        sample_size_option = st.selectbox(
            "Dataset Sample Size",
            options=["1,000", "5,000", "10,000", "25,000", "50,000", "100,000", "250,000", "500,000", "Full Dataset (~360k)"],
            index=st.session_state.sample_size_index,
            help="Number of conversations to load. Higher values provide more data but may slow down the app.",
            key="sample_size_selector_viz"
        )

        # Update session state based on selection
        options_list = ["1,000", "5,000", "10,000", "25,000", "50,000", "100,000", "250,000", "500,000", "Full Dataset (~360k)"]
        st.session_state.sample_size_index = options_list.index(sample_size_option)

        # Convert option to sample size
        if "Full Dataset" in sample_size_option:
            sample_size = None  # Will load entire dataset
            st.warning("⚠️ Loading the full dataset (~360k conversations) may take 2-3 minutes and use significant memory.")
        else:
            sample_size = int(sample_size_option.replace(",", ""))

        st.markdown("---")

        # Dataset links
        st.markdown("### 📂 Access Raw Datasets")
        st.link_button(
            "📊 Conversations Dataset",
            "https://huggingface.co/datasets/ministere-culture/comparia-conversations",
            use_container_width=True
        )
        st.link_button(
            "🗳️ Votes Dataset",
            "https://huggingface.co/datasets/ministere-culture/comparia-votes",
            use_container_width=True
        )
        st.link_button(
            "👍 Reactions Dataset",
            "https://huggingface.co/datasets/ministere-culture/comparia-reactions",
            use_container_width=True
        )

        st.markdown("---")
        st.image("english-logo.png", use_column_width=True)

    # Title
    st.markdown('<div class="main-title">Explore compar:IA datasets</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dataset visualizations and statistics</div>', unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading data..."):
        df = load_conversations(sample_size)

    # Basic statistics
    st.markdown("### 📈 Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Conversations", f"{len(df):,}")

    with col2:
        unique_models = len(set(df['model_a_name'].unique()) | set(df['model_b_name'].unique()))
        st.metric("Unique Models", unique_models)

    with col3:
        avg_turns = df['conv_turns'].mean() if 'conv_turns' in df.columns else 0
        st.metric("Avg Conversation Turns", f"{avg_turns:.1f}")

    # Get all categories for later use
    all_categories = []
    for cat_list in df['categories']:
        if cat_list is not None and hasattr(cat_list, '__iter__') and len(cat_list) > 0:
            cats = list(cat_list) if not isinstance(cat_list, list) else cat_list
            all_categories.extend(cats)

    st.markdown("---")

    # Categories distribution
    st.markdown("### 🏷️ Top Categories")
    if all_categories:
        cat_counts = pd.Series(all_categories).value_counts().head(15)
        st.bar_chart(cat_counts)

    st.markdown("---")

    # Model Lifecycle Timeline
    st.markdown("### ⏱️ Model Lifecycle Timeline")

    # Calculate model lifespans
    with st.spinner("Analyzing model lifecycles..."):
        # Get unique models from both columns
        all_models_a = df.groupby('model_a_name')['timestamp'].agg(['min', 'max', 'count']).reset_index()
        all_models_a.columns = ['model', 'first', 'last', 'count_a']

        all_models_b = df.groupby('model_b_name')['timestamp'].agg(['min', 'max', 'count']).reset_index()
        all_models_b.columns = ['model', 'first', 'last', 'count_b']

        # Merge to get complete timeline
        timeline = all_models_a.merge(all_models_b, on='model', how='outer', suffixes=('_a', '_b'))

        # Calculate true first and last dates
        timeline['first'] = timeline[['first_a', 'first_b']].min(axis=1)
        timeline['last'] = timeline[['last_a', 'last_b']].max(axis=1)
        timeline['total'] = timeline['count_a'].fillna(0) + timeline['count_b'].fillna(0)

        # Keep only necessary columns
        timeline = timeline[['model', 'first', 'last', 'total']]
        timeline['duration_days'] = (timeline['last'] - timeline['first']).dt.days

        # Sort chronologically by first appearance
        timeline_sorted = timeline.sort_values('first')
        color_scheme = 'viridis'
        title = f'Model Lifespan Timeline - All {len(timeline)} Models (Chronological)'

        # Create figure
        fig, ax = plt.subplots(figsize=(18, len(timeline_sorted) * 0.35 + 2))

        # Create color mapping
        if color_scheme == 'viridis':
            norm = plt.Normalize(vmin=np.log1p(timeline_sorted['total'].min()),
                                vmax=np.log1p(timeline_sorted['total'].max()))
            colors = plt.cm.viridis(norm(np.log1p(timeline_sorted['total'])))
        else:
            colors = [plt.cm.coolwarm(i / len(timeline_sorted)) for i in range(len(timeline_sorted))]

        # Plot each model
        for i, (idx, row) in enumerate(timeline_sorted.iterrows()):
            duration = (row['last'] - row['first']).days

            # Main timeline bar
            ax.barh(i, duration, left=row['first'], height=0.5,
                   color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.3)

            # Add start marker (green circle)
            ax.scatter(row['first'], i, s=25, color='green', zorder=5,
                      edgecolor='darkgreen', linewidth=0.5, alpha=0.8)

            # Add end marker (red square)
            ax.scatter(row['last'], i, s=25, color='red', marker='s', zorder=5,
                      edgecolor='darkred', linewidth=0.5, alpha=0.8)

            # Add model name on the left
            ax.text(timeline_sorted['first'].min() - pd.Timedelta(days=3), i,
                   row['model'][:30], ha='right', va='center', fontsize=7)

            # Add usage count on the right
            ax.text(timeline_sorted['last'].max() + pd.Timedelta(days=3), i,
                   f"{int(row['total']):,}", ha='left', va='center',
                   fontsize=6, color='#666666')

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Set limits
        ax.set_ylim(-1, len(timeline_sorted))
        ax.set_xlim(timeline_sorted['first'].min() - pd.Timedelta(days=7),
                   timeline_sorted['last'].max() + pd.Timedelta(days=30))
        ax.set_yticks([])

        # Add title and labels
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        ax.set_xlabel('Timeline', fontsize=12)

        # Add legend
        from matplotlib.lines import Line2D
        green_marker = Line2D([0], [0], marker='o', color='w',
                             markerfacecolor='green', markersize=6,
                             label='First conversation')
        red_marker = Line2D([0], [0], marker='s', color='w',
                           markerfacecolor='red', markersize=6,
                           label='Last conversation')
        ax.legend(handles=[green_marker, red_marker], loc='upper right',
                 frameon=True, fancybox=True, shadow=True, fontsize=9)

        # Add grid
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Add usage count label
        ax.text(timeline_sorted['last'].max() + pd.Timedelta(days=3), -0.5,
               'Total', ha='left', va='center', fontsize=7,
               style='italic', color='#666666')

        plt.tight_layout()

        # Display in Streamlit
        st.pyplot(fig)
        plt.close()

    # Summary statistics
    st.markdown("#### 📊 Summary Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Unique Models", len(timeline))
        st.caption(f"From {timeline['first'].min().date()} to {timeline['last'].max().date()}")

    with col2:
        avg_duration = timeline['duration_days'].mean()
        st.metric("Avg Model Lifespan", f"{avg_duration:.0f} days")

    with col3:
        median_usage = timeline['total'].median()
        st.metric("Median Usage", f"{int(median_usage):,} conversations")

    # Top models table
    st.markdown("#### 🏆 Top 10 Most Used Models")
    top_models = timeline.nlargest(10, 'total')[['model', 'total', 'duration_days', 'first', 'last']].copy()
    top_models.columns = ['Model', 'Total Conversations', 'Days Active', 'First Seen', 'Last Seen']
    top_models['Total Conversations'] = top_models['Total Conversations'].astype(int)
    top_models['First Seen'] = top_models['First Seen'].dt.date
    top_models['Last Seen'] = top_models['Last Seen'].dt.date
    st.dataframe(top_models, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Votes and Reactions Analysis
    st.markdown("### 🗳️ Votes & Reactions Analysis")

    with st.spinner("Loading votes and reactions data..."):
        df_votes = load_votes()

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_votes = len(df_votes)
        st.metric("Total Votes & Reactions", f"{total_votes:,}")

    with col2:
        votes_only = len(df_votes[df_votes['source'] == 'votes'])
        st.metric("Votes", f"{votes_only:,}")

    with col3:
        reactions_only = len(df_votes[df_votes['source'] == 'reactions'])
        st.metric("Reactions", f"{reactions_only:,}")

    with col4:
        tie_rate = (df_votes['both_equal'].sum() / len(df_votes) * 100) if len(df_votes) > 0 else 0
        st.metric("Tie Rate", f"{tie_rate:.1f}%")

    st.markdown("#### 📊 Vote Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Outcome distribution
        st.markdown("**Outcome Distribution**")
        outcome_counts = pd.Series({
            'Model Chosen': len(df_votes[df_votes['both_equal'] == False]),
            'Tie (Both Equal)': df_votes['both_equal'].sum()
        })
        st.bar_chart(outcome_counts)

    with col2:
        # Source distribution
        st.markdown("**Data Source**")
        source_counts = df_votes['source'].value_counts()
        st.bar_chart(source_counts)

    st.markdown("#### 🏅 Model Win Rates")

    # Calculate win rates per model
    wins_by_model = df_votes[df_votes['both_equal'] == False]['chosen_model_name'].value_counts().head(15)

    if len(wins_by_model) > 0:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.bar_chart(wins_by_model)

        with col2:
            st.dataframe(
                pd.DataFrame({
                    'Model': wins_by_model.index,
                    'Wins': wins_by_model.values
                }),
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("No model win data available")

if __name__ == "__main__":
    main()
