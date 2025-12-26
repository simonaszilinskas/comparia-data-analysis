# compar:IA Conversations Explorer

An interactive Streamlit application for exploring and analyzing conversations from the [compar:IA](https://comparia.ai) model comparison platform.

## Dataset

This app uses the `ministere-culture/comparia-conversations` dataset from Hugging Face, containing over 360,000 conversations comparing 91+ different AI models.

## Features

The app has two main tabs accessible from the top of the page:

### 🔍 Search Tab
- **Search conversations**: Find conversations by keywords in content, summaries, and messages
- **Model filtering**: Filter by specific AI models
- **Category filtering**: Browse conversations by topic categories
- **Vote filtering**: Show only conversations with user preference votes
- **Side-by-side comparison**: View Model A and Model B responses with vote indicators
- **Pagination**: Browse through search results efficiently

### 📊 Visualizations Tab
- **Dataset statistics**: Total conversations, unique models, categories, and average turns
- **Top models**: Most frequently used AI models in conversations
- **Category breakdown**: Distribution of conversation topics
- **Interactive charts**: Visual exploration of the dataset

## Installation

1. Clone or navigate to this directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. The app will automatically download the dataset on first run (cached for subsequent runs)

3. Navigate between tabs at the top of the page:
   - **🔍 Search**: Search and browse conversations with filters
   - **📊 Visualizations**: Explore dataset statistics and charts

4. In the Search tab, use filters in the sidebar to refine results

5. Click on conversations to expand and view details side-by-side

## Dataset Statistics

- **Total conversations**: 360,088
- **Date range**: October 2024 - December 2024
- **Models**: 91 unique AI models
- **Languages**: 130+ languages (94.6% French, 8.9% English)
- **Top categories**: Natural Science & Technology (38%), Education (31%), Business & Economics (22%)
- **Total energy consumed**: ~26,900 kWh

## Performance Notes

- The dataset is cached after first load for better performance
- Filtering and search operations are optimized for responsiveness
- Large result sets are paginated for better UX

## For Researchers

This tool is designed to help researchers:
- Explore conversation patterns across different models
- Analyze energy consumption and efficiency metrics
- Study topic distribution and language usage
- Export filtered datasets for further analysis
- Understand model behavior in different contexts

## License

This project uses the compar:IA dataset which is publicly available on Hugging Face.
