import streamlit as st
import pandas as pd
from datasets import load_dataset
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from collections import Counter

# Page config
st.set_page_config(
    page_title="compar:IA Conversations Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stat-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #0068c9;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
    .conversation-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    .model-tag {
        background-color: #0068c9;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Cache the dataset loading
@st.cache_data
def load_data(sample_size=None):
    """Load the dataset and convert to pandas

    Args:
        sample_size: Number of rows to load, None for all data
    """
    if sample_size:
        ds = load_dataset('ministere-culture/comparia-conversations', split=f'train[:{sample_size}]')
    else:
        ds = load_dataset('ministere-culture/comparia-conversations', split='train')

    df = ds.to_pandas()

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df

@st.cache_data
def get_unique_values(df):
    """Extract unique values for filters"""
    models = sorted(set(df['model_a_name'].unique()) | set(df['model_b_name'].unique()))

    # Extract all categories
    all_categories = []
    for cat_list in df['categories']:
        if cat_list is not None and isinstance(cat_list, list) and len(cat_list) > 0:
            all_categories.extend(cat_list)
    categories = sorted(set(all_categories)) if all_categories else []

    # Extract all languages
    all_languages = []
    for lang_list in df['languages']:
        if lang_list is not None and isinstance(lang_list, list) and len(lang_list) > 0:
            all_languages.extend(lang_list)
    languages = sorted(set(all_languages)) if all_languages else []

    # Extract modes and filter out None values
    modes_unique = df['mode'].unique()
    modes = sorted([m for m in modes_unique if m is not None])

    return models, categories, languages, modes

def filter_dataframe(df, filters):
    """Apply all filters to the dataframe"""
    filtered_df = df.copy()

    # Text search in conversations
    if filters['text_search']:
        search_term = filters['text_search'].lower()
        mask = filtered_df.apply(lambda row: search_in_conversation(row, search_term), axis=1)
        filtered_df = filtered_df[mask]

    # Model filters
    if filters['models_a']:
        filtered_df = filtered_df[filtered_df['model_a_name'].isin(filters['models_a'])]
    if filters['models_b']:
        filtered_df = filtered_df[filtered_df['model_b_name'].isin(filters['models_b'])]

    # Category filter
    if filters['categories']:
        mask = filtered_df['categories'].apply(
            lambda cats: any(cat in filters['categories'] for cat in cats) if (cats is not None and isinstance(cats, list)) else False
        )
        filtered_df = filtered_df[mask]

    # Language filter
    if filters['languages']:
        mask = filtered_df['languages'].apply(
            lambda langs: any(lang in filters['languages'] for lang in langs) if (langs is not None and isinstance(langs, list)) else False
        )
        filtered_df = filtered_df[mask]

    # Mode filter
    if filters['modes']:
        filtered_df = filtered_df[filtered_df['mode'].isin(filters['modes'])]

    # Date range filter
    if filters['date_range']:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['timestamp'] >= pd.Timestamp(start_date)) &
            (filtered_df['timestamp'] <= pd.Timestamp(end_date))
        ]

    # Conversation turns filter
    if filters['turns_range']:
        min_turns, max_turns = filters['turns_range']
        filtered_df = filtered_df[
            (filtered_df['conv_turns'] >= min_turns) &
            (filtered_df['conv_turns'] <= max_turns)
        ]

    # Keyword search
    if filters['keyword_search']:
        keyword_term = filters['keyword_search'].lower()
        mask = filtered_df['keywords'].apply(
            lambda kws: any(keyword_term in kw.lower() for kw in kws) if (kws is not None and isinstance(kws, list)) else False
        )
        filtered_df = filtered_df[mask]

    return filtered_df

def search_in_conversation(row, search_term):
    """Search for text in conversation content and summary"""
    # Search in summary
    if row['short_summary'] and isinstance(row['short_summary'], str) and search_term in row['short_summary'].lower():
        return True

    # Search in opening message
    if row['opening_msg'] and isinstance(row['opening_msg'], str) and search_term in row['opening_msg'].lower():
        return True

    # Search in conversations
    for conv_list in [row['conversation_a'], row['conversation_b']]:
        if conv_list is not None and isinstance(conv_list, list) and len(conv_list) > 0:
            for turn in conv_list:
                if turn and isinstance(turn, dict) and turn.get('content') and search_term in turn['content'].lower():
                    return True

    return False

def display_stats_cards(df):
    """Display key statistics as cards"""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(df):,}</div>
            <div class="stat-label">Conversations</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        unique_models = len(set(df['model_a_name'].unique()) | set(df['model_b_name'].unique()))
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{unique_models}</div>
            <div class="stat-label">Unique Models</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_turns = df['conv_turns'].mean()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{avg_turns:.1f}</div>
            <div class="stat-label">Avg Turns</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_energy = df['total_conv_a_kwh'].sum() + df['total_conv_b_kwh'].sum()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_energy:.1f}</div>
            <div class="stat-label">Total kWh</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        avg_tokens = (df['total_conv_a_output_tokens'].mean() + df['total_conv_b_output_tokens'].mean()) / 2
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{avg_tokens:.0f}</div>
            <div class="stat-label">Avg Tokens</div>
        </div>
        """, unsafe_allow_html=True)

def create_visualizations(df):
    """Create dynamic visualizations based on filtered data"""

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 Models", "📚 Topics", "⚡ Energy"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Conversation turns distribution
            fig_turns = px.histogram(
                df,
                x='conv_turns',
                title='Conversation Length Distribution',
                labels={'conv_turns': 'Number of Turns', 'count': 'Frequency'},
                nbins=50
            )
            fig_turns.update_layout(height=350)
            st.plotly_chart(fig_turns, use_container_width=True)

        with col2:
            # Mode distribution
            mode_counts = df['mode'].value_counts()
            fig_mode = px.pie(
                values=mode_counts.values,
                names=mode_counts.index,
                title='Comparison Mode Distribution'
            )
            fig_mode.update_layout(height=350)
            st.plotly_chart(fig_mode, use_container_width=True)

        # Timeline
        df_timeline = df.groupby(df['timestamp'].dt.date).size().reset_index()
        df_timeline.columns = ['date', 'count']
        fig_timeline = px.line(
            df_timeline,
            x='date',
            y='count',
            title='Conversations Over Time',
            labels={'date': 'Date', 'count': 'Number of Conversations'}
        )
        fig_timeline.update_layout(height=300)
        st.plotly_chart(fig_timeline, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            # Top models
            model_counts = pd.concat([
                df['model_a_name'].value_counts(),
                df['model_b_name'].value_counts()
            ]).groupby(level=0).sum().sort_values(ascending=False).head(15)

            fig_models = px.bar(
                x=model_counts.values,
                y=model_counts.index,
                orientation='h',
                title='Top 15 Models by Frequency',
                labels={'x': 'Appearances', 'y': 'Model'}
            )
            fig_models.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_models, use_container_width=True)

        with col2:
            # Model efficiency (tokens per kWh)
            model_efficiency = []
            for model in df['model_a_name'].unique():
                df_model_a = df[df['model_a_name'] == model]
                df_model_b = df[df['model_b_name'] == model]

                total_tokens = (
                    df_model_a['total_conv_a_output_tokens'].sum() +
                    df_model_b['total_conv_b_output_tokens'].sum()
                )
                total_energy = (
                    df_model_a['total_conv_a_kwh'].sum() +
                    df_model_b['total_conv_b_kwh'].sum()
                )

                if total_energy > 0:
                    efficiency = total_tokens / total_energy
                    appearances = len(df_model_a) + len(df_model_b)
                    model_efficiency.append({
                        'model': model,
                        'tokens_per_kwh': efficiency,
                        'appearances': appearances
                    })

            df_efficiency = pd.DataFrame(model_efficiency).sort_values('tokens_per_kwh', ascending=False).head(15)

            fig_efficiency = px.bar(
                df_efficiency,
                x='tokens_per_kwh',
                y='model',
                orientation='h',
                title='Top 15 Most Energy-Efficient Models (Tokens/kWh)',
                labels={'tokens_per_kwh': 'Tokens per kWh', 'model': 'Model'}
            )
            fig_efficiency.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_efficiency, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            # Category distribution
            all_categories = []
            for cat_list in df['categories']:
                if cat_list is not None and isinstance(cat_list, list) and len(cat_list) > 0:
                    all_categories.extend(cat_list)

            cat_counts = pd.Series(all_categories).value_counts().head(15) if all_categories else pd.Series()

            fig_categories = px.bar(
                x=cat_counts.values,
                y=cat_counts.index,
                orientation='h',
                title='Top 15 Categories',
                labels={'x': 'Count', 'y': 'Category'}
            )
            fig_categories.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_categories, use_container_width=True)

        with col2:
            # Language distribution
            all_languages = []
            for lang_list in df['languages']:
                if lang_list is not None and isinstance(lang_list, list) and len(lang_list) > 0:
                    all_languages.extend(lang_list)

            lang_counts = pd.Series(all_languages).value_counts().head(15) if all_languages else pd.Series()

            fig_languages = px.bar(
                x=lang_counts.values,
                y=lang_counts.index,
                orientation='h',
                title='Top 15 Languages',
                labels={'x': 'Count', 'y': 'Language'}
            )
            fig_languages.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_languages, use_container_width=True)

        # Top keywords
        all_keywords = []
        for kw_list in df['keywords']:
            if kw_list is not None and isinstance(kw_list, list) and len(kw_list) > 0:
                all_keywords.extend(kw_list)

        kw_counts = pd.Series(all_keywords).value_counts().head(30)

        fig_keywords = px.bar(
            x=kw_counts.values,
            y=kw_counts.index,
            orientation='h',
            title='Top 30 Keywords',
            labels={'x': 'Count', 'y': 'Keyword'}
        )
        fig_keywords.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_keywords, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            # Energy by category
            category_energy = {}
            for _, row in df.iterrows():
                if row['categories'] is not None and isinstance(row['categories'], list) and len(row['categories']) > 0:
                    energy = row['total_conv_a_kwh'] + row['total_conv_b_kwh']
                    for cat in row['categories']:
                        if cat not in category_energy:
                            category_energy[cat] = 0
                        category_energy[cat] += energy

            df_cat_energy = pd.DataFrame(
                list(category_energy.items()),
                columns=['category', 'energy']
            ).sort_values('energy', ascending=False).head(15)

            fig_cat_energy = px.bar(
                df_cat_energy,
                x='energy',
                y='category',
                orientation='h',
                title='Top 15 Categories by Energy Consumption',
                labels={'energy': 'Total kWh', 'category': 'Category'}
            )
            fig_cat_energy.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_cat_energy, use_container_width=True)

        with col2:
            # Energy over time
            df_energy_time = df.groupby(df['timestamp'].dt.date).agg({
                'total_conv_a_kwh': 'sum',
                'total_conv_b_kwh': 'sum'
            }).reset_index()
            df_energy_time['total_energy'] = (
                df_energy_time['total_conv_a_kwh'] +
                df_energy_time['total_conv_b_kwh']
            )
            df_energy_time.columns = ['date', 'model_a', 'model_b', 'total']

            fig_energy_time = px.line(
                df_energy_time,
                x='date',
                y='total',
                title='Energy Consumption Over Time',
                labels={'date': 'Date', 'total': 'Total kWh'}
            )
            fig_energy_time.update_layout(height=500)
            st.plotly_chart(fig_energy_time, use_container_width=True)

def display_conversation(row):
    """Display a single conversation in an expandable format"""
    with st.expander(f"💬 {row['short_summary'][:100]}..." if row['short_summary'] else f"💬 Conversation {row['id']}"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Model A:** `{row['model_a_name']}`")
            st.markdown(f"**Tokens:** {row['total_conv_a_output_tokens']:,}")
            st.markdown(f"**Energy:** {row['total_conv_a_kwh']:.6f} kWh")

        with col2:
            st.markdown(f"**Model B:** `{row['model_b_name']}`")
            st.markdown(f"**Tokens:** {row['total_conv_b_output_tokens']:,}")
            st.markdown(f"**Energy:** {row['total_conv_b_kwh']:.6f} kWh")

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"**📅 Date:** {row['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**🔄 Turns:** {row['conv_turns']}")
            st.markdown(f"**🎯 Mode:** {row['mode']}")

        with col2:
            if row['languages']:
                st.markdown(f"**🌐 Languages:** {', '.join(row['languages'])}")
            if row['categories']:
                st.markdown(f"**📚 Categories:** {', '.join(row['categories'][:3])}")
            if row['keywords']:
                st.markdown(f"**🏷️ Keywords:** {', '.join(row['keywords'][:5])}")

        if row['short_summary']:
            st.markdown("**📝 Summary:**")
            st.info(row['short_summary'])

        # Display conversations side by side
        st.markdown("**💬 Conversation:**")

        conv_a = row['conversation_a'] if row['conversation_a'] else []
        conv_b = row['conversation_b'] if row['conversation_b'] else []

        max_turns = max(len(conv_a), len(conv_b))

        for i in range(max_turns):
            col1, col2 = st.columns(2)

            with col1:
                if i < len(conv_a) and conv_a[i]:
                    role = conv_a[i].get('role', 'unknown')
                    content = conv_a[i].get('content', '')

                    if role == 'user':
                        st.markdown(f"**👤 User:**")
                        st.markdown(f">{content[:500]}..." if len(content) > 500 else f">{content}")
                    else:
                        st.markdown(f"**🤖 Model A:**")
                        st.markdown(content[:500] + "..." if len(content) > 500 else content)

            with col2:
                if i < len(conv_b) and conv_b[i]:
                    role = conv_b[i].get('role', 'unknown')
                    content = conv_b[i].get('content', '')

                    if role == 'user':
                        st.markdown(f"**👤 User:**")
                        st.markdown(f">{content[:500]}..." if len(content) > 500 else f">{content}")
                    else:
                        st.markdown(f"**🤖 Model B:**")
                        st.markdown(content[:500] + "..." if len(content) > 500 else content)

def main():
    st.markdown('<div class="main-header">📊 compar:IA Conversations Explorer</div>', unsafe_allow_html=True)
    st.markdown("Explore and analyze conversations from the compar:IA model comparison platform")

    # Sidebar - Data size selector (at the very top)
    st.sidebar.header("⚙️ Settings")

    data_size_options = {
        "10,000 rows (Fast)": 10000,
        "50,000 rows (Balanced)": 50000,
        "100,000 rows (Large)": 100000,
        "All data (~360K rows)": None
    }

    selected_size = st.sidebar.selectbox(
        "Dataset size",
        options=list(data_size_options.keys()),
        index=0,
        help="Choose how many conversations to load. Smaller datasets load faster."
    )

    sample_size = data_size_options[selected_size]

    if sample_size:
        st.sidebar.info(f"📊 Loading {sample_size:,} conversations for faster performance")
    else:
        st.sidebar.warning("⚠️ Loading full dataset may take 1-2 minutes")

    st.sidebar.markdown("---")

    # Load data
    with st.spinner(f"Loading dataset ({selected_size})..."):
        df = load_data(sample_size)
        models, categories, languages, modes = get_unique_values(df)

    # Sidebar filters
    st.sidebar.header("🔍 Search & Filters")

    filters = {}

    # Text search
    filters['text_search'] = st.sidebar.text_input(
        "Search in conversations",
        placeholder="Enter search term...",
        help="Search in conversation content and summaries"
    )

    # Keyword search
    filters['keyword_search'] = st.sidebar.text_input(
        "Search in keywords",
        placeholder="Enter keyword...",
        help="Search for specific keywords"
    )

    st.sidebar.markdown("---")

    # Model filters
    with st.sidebar.expander("🤖 Models", expanded=False):
        filters['models_a'] = st.multiselect(
            "Model A",
            options=models,
            default=None,
            help="Filter by Model A"
        )
        filters['models_b'] = st.multiselect(
            "Model B",
            options=models,
            default=None,
            help="Filter by Model B"
        )

    # Category filter
    with st.sidebar.expander("📚 Categories", expanded=False):
        filters['categories'] = st.multiselect(
            "Select categories",
            options=categories,
            default=None,
            help="Filter by conversation category"
        )

    # Language filter
    with st.sidebar.expander("🌐 Languages", expanded=False):
        filters['languages'] = st.multiselect(
            "Select languages",
            options=languages,
            default=None,
            help="Filter by conversation language"
        )

    # Mode filter
    with st.sidebar.expander("🎯 Comparison Mode", expanded=False):
        filters['modes'] = st.multiselect(
            "Select modes",
            options=modes,
            default=None,
            help="Filter by comparison mode"
        )

    # Date range filter
    with st.sidebar.expander("📅 Date Range", expanded=False):
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()

        date_range = st.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter by conversation date"
        )

        if len(date_range) == 2:
            filters['date_range'] = date_range
        else:
            filters['date_range'] = None

    # Conversation turns filter
    with st.sidebar.expander("🔄 Conversation Turns", expanded=False):
        min_turns = int(df['conv_turns'].min())
        max_turns = int(df['conv_turns'].max())

        turns_range = st.slider(
            "Number of turns",
            min_value=min_turns,
            max_value=min(max_turns, 20),  # Cap at 20 for UI purposes
            value=(min_turns, min(max_turns, 20)),
            help="Filter by number of conversation turns"
        )
        filters['turns_range'] = turns_range

    # Reset filters button
    if st.sidebar.button("🔄 Reset All Filters"):
        st.rerun()

    # Apply filters
    filtered_df = filter_dataframe(df, filters)

    # Display results
    st.markdown("---")

    # Stats cards
    display_stats_cards(filtered_df)

    st.markdown("---")

    # Visualizations
    if len(filtered_df) > 0:
        create_visualizations(filtered_df)

        st.markdown("---")

        # Results table
        st.header("📋 Conversation Results")

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown(f"**{len(filtered_df):,}** conversations found")

        with col2:
            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                options=['timestamp', 'conv_turns', 'total_conv_a_output_tokens', 'total_conv_a_kwh'],
                format_func=lambda x: {
                    'timestamp': 'Date',
                    'conv_turns': 'Turns',
                    'total_conv_a_output_tokens': 'Tokens',
                    'total_conv_a_kwh': 'Energy'
                }[x]
            )

        with col3:
            sort_order = st.selectbox("Order", options=['Descending', 'Ascending'])

        # Sort dataframe
        sorted_df = filtered_df.sort_values(
            by=sort_by,
            ascending=(sort_order == 'Ascending')
        )

        # Pagination
        items_per_page = st.slider("Conversations per page", min_value=5, max_value=50, value=10, step=5)

        total_pages = max(1, len(sorted_df) // items_per_page + (1 if len(sorted_df) % items_per_page > 0 else 0))
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(sorted_df))

        # Display conversations
        for idx in range(start_idx, end_idx):
            row = sorted_df.iloc[idx]
            display_conversation(row)

        st.markdown(f"Showing {start_idx + 1}-{end_idx} of {len(sorted_df):,} conversations")

        # Export functionality
        st.markdown("---")
        st.header("📥 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            # Export filtered results as CSV
            csv_data = filtered_df[[
                'id', 'timestamp', 'model_a_name', 'model_b_name',
                'conv_turns', 'mode', 'short_summary',
                'total_conv_a_output_tokens', 'total_conv_b_output_tokens',
                'total_conv_a_kwh', 'total_conv_b_kwh'
            ]].to_csv(index=False)

            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"comparia_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            # Export as JSON
            json_data = filtered_df.to_json(orient='records', date_format='iso')

            st.download_button(
                label="📄 Download as JSON",
                data=json_data,
                file_name=f"comparia_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    else:
        st.warning("No conversations found with the current filters. Try adjusting your search criteria.")

if __name__ == "__main__":
    main()
