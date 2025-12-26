import streamlit as st
import pandas as pd
from datasets import load_dataset
from datetime import datetime

# Page config
st.set_page_config(
    page_title="compar:IA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_conversations(sample_size=10000):
    """Load conversations dataset"""
    ds = load_dataset('ministere-culture/comparia-conversations', split=f'train[:{sample_size}]')
    df = ds.to_pandas()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_data
def load_votes():
    """Load votes dataset with user preferences"""
    ds = load_dataset('ministere-culture/comparia-votes', split='train')
    df = ds.to_pandas()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Keep only relevant columns
    return df[['conversation_pair_id', 'chosen_model_name', 'both_equal']]

def search_conversations(df, search_term):
    """Search for conversations containing the search term"""
    if not search_term:
        return pd.DataFrame()

    search_term = search_term.lower()
    matching_indices = []

    for idx, row in df.iterrows():
        found = False

        # Search in summary
        if row['short_summary'] and isinstance(row['short_summary'], str):
            if search_term in row['short_summary'].lower():
                found = True

        # Search in opening message
        if not found and row['opening_msg'] and isinstance(row['opening_msg'], str):
            if search_term in row['opening_msg'].lower():
                found = True

        # Search in conversations
        if not found:
            for conv_list in [row['conversation_a'], row['conversation_b']]:
                if conv_list is not None and hasattr(conv_list, '__iter__') and len(conv_list) > 0:
                    for turn in conv_list:
                        if turn and isinstance(turn, dict) and turn.get('content'):
                            if search_term in turn['content'].lower():
                                found = True
                                break
                if found:
                    break

        if found:
            matching_indices.append(idx)

    return df.loc[matching_indices] if matching_indices else pd.DataFrame()

def display_conversation(row, votes_df, result_num=None):
    """Display a single conversation with user preference"""

    # Get user preference if available
    preference = None
    if len(votes_df) > 0:
        vote_match = votes_df[votes_df['conversation_pair_id'] == row['conversation_pair_id']]
        if len(vote_match) > 0:
            vote = vote_match.iloc[0]
            if vote['both_equal']:
                preference = 'tie'
            elif vote['chosen_model_name']:
                preference = vote['chosen_model_name']

    # Get conversations first to extract the prompt
    conv_a = list(row['conversation_a']) if (row['conversation_a'] is not None and hasattr(row['conversation_a'], '__iter__')) else []
    conv_b = list(row['conversation_b']) if (row['conversation_b'] is not None and hasattr(row['conversation_b'], '__iter__')) else []

    # Get the first user message as the title
    title = "Conversation"
    if len(conv_a) > 0 and isinstance(conv_a[0], dict) and conv_a[0].get('role') == 'user':
        prompt = conv_a[0].get('content', '')
        # Truncate if too long
        if len(prompt) > 100:
            title = prompt[:100] + "..."
        else:
            title = prompt

    # Add vote indicator to title
    if preference == 'tie':
        vote_indicator = " [Both Equal]"
    elif preference == row['model_a_name']:
        vote_indicator = f" [✅ {row['model_a_name']}]"
    elif preference == row['model_b_name']:
        vote_indicator = f" [✅ {row['model_b_name']}]"
    else:
        vote_indicator = ""

    title = title + vote_indicator

    # Use expander with the prompt as title
    with st.expander(title, expanded=False):
        if len(conv_a) == 0 and len(conv_b) == 0:
            st.warning("No conversation data available")
        else:
            # Display each turn (pair of user message + assistant responses)
            # Calculate number of turns based on user messages
            num_turns_a = len([m for m in conv_a if isinstance(m, dict) and m.get('role') == 'user'])
            num_turns_b = len([m for m in conv_b if isinstance(m, dict) and m.get('role') == 'user'])
            max_turns = max(num_turns_a, num_turns_b)

            for turn_idx in range(max_turns):
                # User and assistant messages are at alternating indices
                user_idx = turn_idx * 2
                assistant_idx = turn_idx * 2 + 1

                # User question
                if user_idx < len(conv_a) and isinstance(conv_a[user_idx], dict) and conv_a[user_idx].get('role') == 'user':
                    user_content = conv_a[user_idx].get('content', '')
                    st.markdown(f"**👤 User Question:**")
                    st.info(user_content)
                    st.markdown("")

                # Model responses side by side
                col1, col2 = st.columns(2)

                # Determine vote indicators for titles
                if preference == 'tie':
                    title_a = f"**Both Equal - Model A Response ({row['model_a_name']}):**"
                    title_b = f"**Both Equal - Model B Response ({row['model_b_name']}):**"
                elif preference == row['model_a_name']:
                    title_a = f"**✅ Winner: Model A Response ({row['model_a_name']}):**"
                    title_b = f"**Model B Response ({row['model_b_name']}):**"
                elif preference == row['model_b_name']:
                    title_a = f"**Model A Response ({row['model_a_name']}):**"
                    title_b = f"**✅ Winner: Model B Response ({row['model_b_name']}):**"
                else:
                    title_a = f"**Model A Response ({row['model_a_name']}):**"
                    title_b = f"**Model B Response ({row['model_b_name']}):**"

                with col1:
                    st.markdown(title_a)
                    if assistant_idx < len(conv_a) and isinstance(conv_a[assistant_idx], dict) and conv_a[assistant_idx].get('role') == 'assistant':
                        response_a = conv_a[assistant_idx].get('content', '')
                        if response_a:
                            st.markdown(response_a)
                        else:
                            st.text("No response")
                    else:
                        st.text("No response")

                with col2:
                    st.markdown(title_b)
                    if assistant_idx < len(conv_b) and isinstance(conv_b[assistant_idx], dict) and conv_b[assistant_idx].get('role') == 'assistant':
                        response_b = conv_b[assistant_idx].get('content', '')
                        if response_b:
                            st.markdown(response_b)
                        else:
                            st.text("No response")
                    else:
                        st.text("No response")

                # Add spacing between turns
                if turn_idx < max_turns - 1:
                    st.markdown("---")

        # Display metadata at the bottom
        st.markdown("---")
        meta_col1, meta_col2, meta_col3 = st.columns(3)

        with meta_col1:
            # Timestamp
            if pd.notna(row['timestamp']):
                timestamp = pd.to_datetime(row['timestamp'])
                st.caption(f"📅 {timestamp.strftime('%Y-%m-%d %H:%M')}")

        with meta_col2:
            # Categories
            if row['categories'] is not None and hasattr(row['categories'], '__iter__') and len(row['categories']) > 0:
                cats = list(row['categories']) if not isinstance(row['categories'], list) else row['categories']
                st.caption(f"🏷️ {', '.join(cats[:3])}")  # Show first 3 categories

        with meta_col3:
            # Conversation ID
            st.caption(f"🆔 {row['conversation_pair_id']}")

def main():
    # Title
    st.markdown('<div class="main-title">🔍 compar:IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore and analyze AI model conversations</div>', unsafe_allow_html=True)

    # Load data
    sample_size = 10000  # Fixed sample size
    with st.spinner("Loading data..."):
        df_conversations = load_conversations(sample_size)
        df_votes = load_votes()

    # Tab navigation at the top
    tab1, tab2 = st.tabs(["🔍 Search", "📊 Visualizations"])

    with tab1:
        # SEARCH TAB
        render_search_tab(df_conversations, df_votes)

    with tab2:
        # VISUALIZATIONS TAB
        render_visualizations_tab(df_conversations)

def render_search_tab(df_conversations, df_votes):
    """Render the search tab"""
    # Filters in sidebar
    with st.sidebar:
        st.header("🔧 Filters")

        # Get unique models
        all_models = sorted(set(df_conversations['model_a_name'].unique()) |
                          set(df_conversations['model_b_name'].unique()))
        selected_models = st.multiselect(
            "Models",
            options=all_models,
            default=[],
            help="Filter conversations by model"
        )

        # Get unique categories
        all_categories = []
        for cat_list in df_conversations['categories']:
            if cat_list is not None and hasattr(cat_list, '__iter__') and len(cat_list) > 0:
                cats = list(cat_list) if not isinstance(cat_list, list) else cat_list
                all_categories.extend(cats)
        unique_categories = sorted(set(all_categories)) if all_categories else []

        selected_categories = st.multiselect(
            "Categories",
            options=unique_categories,
            default=[],
            help="Filter conversations by category"
        )

        show_only_voted = st.checkbox("Only show voted conversations", value=False, help="Show only conversations with user votes")

    # Initialize active search in session state if not exists
    if 'active_search' not in st.session_state:
        st.session_state.active_search = ''

    # Main search box with button
    with st.form(key="search_form"):
        col1, col2 = st.columns([6, 1])
        with col1:
            search_term = st.text_input(
                "",
                value=st.session_state.active_search,
                placeholder="Enter keywords to search in conversations (e.g., 'Python', 'recipe', 'mathematics')...",
                key="search",
                label_visibility="collapsed"
            )
        with col2:
            search_button = st.form_submit_button("🔍 Search", use_container_width=True)

    # Update search term in session state when search is triggered
    if search_button:
        st.session_state.active_search = search_term

    # Get the active search term
    active_search = st.session_state.active_search

    # Search (triggered by button or Enter)
    if active_search:
        with st.spinner("Searching..."):
            results = search_conversations(df_conversations, active_search)

            # Apply filters
            if len(results) > 0:
                if selected_models:
                    results = results[
                        results['model_a_name'].isin(selected_models) |
                        results['model_b_name'].isin(selected_models)
                    ]

                if selected_categories:
                    results = results[
                        results['categories'].apply(
                            lambda cats: any(cat in selected_categories for cat in (list(cats) if hasattr(cats, '__iter__') else []))
                            if cats is not None else False
                        )
                    ]

                if show_only_voted:
                    results = results[
                        results['conversation_pair_id'].isin(df_votes['conversation_pair_id'])
                    ]

            # Display results
            if len(results) > 0:
                st.success(f"Found **{len(results)}** conversation(s) matching your search")

                # Pagination
                items_per_page = 50
                total_pages = max(1, len(results) // items_per_page + (1 if len(results) % items_per_page > 0 else 0))

                # Page selector at top if more than one page
                page = 1
                if total_pages > 1:
                    st.markdown(f"**Total: {len(results)} results ({total_pages} pages)**")

                # Use session state to track page
                if 'search_page' not in st.session_state:
                    st.session_state.search_page = 1

                page = st.session_state.search_page

                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(results))

                # Display conversations
                for idx in range(start_idx, end_idx):
                    result_num = idx + 1
                    display_conversation(results.iloc[idx], df_votes, result_num)

                # Page selector at bottom
                if total_pages > 1:
                    st.markdown("---")
                    new_page = st.number_input("Page", min_value=1, max_value=total_pages, value=page, key="page_selector")
                    if new_page != page:
                        st.session_state.search_page = new_page
                        st.rerun()
                    st.markdown(f"Showing {start_idx + 1}-{end_idx} of {len(results)} results")
            else:
                st.warning("No conversations found matching your search. Try different keywords or adjust filters.")
    else:
        # Show random conversations when no search term
        st.info("💡 Showing random conversations from the dataset. Enter keywords to search.")

        # Apply filters first to the entire dataset
        filtered_df = df_conversations.copy()

        if selected_models:
            filtered_df = filtered_df[
                filtered_df['model_a_name'].isin(selected_models) |
                filtered_df['model_b_name'].isin(selected_models)
            ]

        if selected_categories:
            filtered_df = filtered_df[
                filtered_df['categories'].apply(
                    lambda cats: any(cat in selected_categories for cat in (list(cats) if hasattr(cats, '__iter__') else []))
                    if cats is not None else False
                )
            ]

        if show_only_voted:
            filtered_df = filtered_df[
                filtered_df['conversation_pair_id'].isin(df_votes['conversation_pair_id'])
            ]

        # Now sample from filtered dataset
        num_random = 50
        if len(filtered_df) > 0:
            results = filtered_df.sample(n=min(num_random, len(filtered_df)))

            # Display random conversations
            for idx, (_, row) in enumerate(results.iterrows(), 1):
                display_conversation(row, df_votes, idx)
        else:
            st.warning("No conversations match the selected filters.")

def render_visualizations_tab(df):
    """Render the visualizations tab"""
    st.markdown("### 📊 Dataset Visualizations")

    # Basic statistics
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
            model_counts.reset_index().rename(columns={'index': 'Model', 0: 'Count'}),
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
