import nltk

# Check if NLTK data is downloaded, if not, download it
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# Streamlit app title
st.title('Sentiment Analysis and Visualization')

data_path = st.file_uploader("Upload CSV file", type=["csv"])
if data_path:
    df = pd.read_csv(data_path)

    # Create a PDF file to save the output
    pdf_path = '/output.pdf'
    pdf_pages = PdfPages(pdf_path)

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Handle missing values in 'review_text' column
    df['review_text'].fillna('', inplace=True)

    # Perform sentiment analysis using the NLTK Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()
    df['compound_sentiment'] = df['review_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

    # Round up ratings to whole numbers
    df['rating'] = np.ceil(df['rating']).astype(int)

    # Add the 'review_length' column
    df['review_length'] = df['review_text'].apply(len)

    # Positive and negative keywords
    positive_keywords = ['good', 'excellent', 'positive', 'satisfactory', 'commendable']
    negative_keywords = ['bad', 'poor', 'worst', 'negative', 'unsatisfactory', 'disappointing']
    suggestion_keywords = ['suggestion', 'improvement', 'recommendation']

    # Categorize Feedback
    df['feedback_category'] = 'review_text'
    df.loc[df['review_text'].str.contains('|'.join(positive_keywords), case=False), 'feedback_category'] = 'Positive'
    df.loc[df['review_text'].str.contains('|'.join(negative_keywords), case=False), 'feedback_category'] = 'Negative'
    df.loc[df['review_text'].str.contains('|'.join(suggestion_keywords), case=False), 'feedback_category'] = 'Suggestion'

    # --- Chart 1: Sentiment Analysis based on Ratings ---
    st.subheader('Chart 1: Sentiment Analysis based on Ratings')
    fig, ax1 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='rating', y='compound_sentiment', data=df, hue='rating', palette='viridis', ax=ax1)
    ax1.set_xticks(list(sorted(df['rating'].unique())))
    ax1.set_xlabel('Rating', fontsize='large')
    ax1.set_ylabel('Compound Sentiment', fontsize='large')
    ax1.set_title('Sentiment Analysis based on Ratings', fontsize='x-large')
    ax1.legend(fontsize='medium', title_fontsize='large')
    st.pyplot(fig)

    # --- Chart 2: Average Sentiment per Business ---
    st.subheader('Chart 2: Average Sentiment per Business')
    fig, ax2 = plt.subplots(figsize=(12, 8))
    sns.barplot(x='business_column', y='compound_sentiment', data=df, ax=ax2, palette='viridis')
    ax2.set_xlabel('Business', fontsize='large')
    ax2.set_ylabel('Average Compound Sentiment', fontsize='large')
    ax2.set_title('Average Sentiment Per Product OR Service', fontsize='x-large')
    st.pyplot(fig)

    # --- Chart 3: Review Length Analysis ---
    st.subheader('Chart 3: Review Length Analysis')
    fig, ax3 = plt.subplots(figsize=(12, 8))
    sns.histplot(df['review_length'], bins=20, color='darkblue', kde=False, ax=ax3)
    ax3.set_xlabel('Review Length', fontsize='large')
    ax3.set_ylabel('Frequency', fontsize='large')
    ax3.set_title('Review Length Analysis', fontsize='x-large')
    st.pyplot(fig)

    # --- Chart 4: Word Cloud - Frequency of Mentions ---
    st.subheader('Chart 4: Word Cloud - Frequency of Mentions')
    fig, ax6 = plt.subplots(figsize=(12, 8))
    wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white',
                          colormap='viridis').generate(' '.join(df['review_text']))
    ax6.imshow(wordcloud, interpolation='bilinear')
    ax6.axis('off')
    ax6.set_title('The Most Visible and Popular Words', fontsize='x-large')
    st.pyplot(fig)

    # --- Chart 5: Feedback Analysis - Distribution of Feedback Categories ---
    st.subheader('Chart 5: Distribution of Feedback Categories')
    fig, ax7 = plt.subplots(figsize=(12, 8))
    sns.countplot(x='feedback_category', data=df, palette='viridis', ax=ax7)
    ax7.set_xlabel('Feedback Category', fontsize='large')
    ax7.set_ylabel('Count', fontsize='large')
    ax7.set_title('Distribution of Feedback Categories', fontsize='x-large')
    ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

    # --- Chart 6: Sentiment Trends based on Time Periods ---
    st.subheader('Chart 6: Sentiment Trends based on Time Periods')
    df['converted_date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='converted_date')
    fig, ax5 = plt.subplots(figsize=(12, 8))
    sns.lineplot(x='converted_date', y='compound_sentiment', data=df, ax=ax5)
    ax5.xaxis.set_major_locator(mdates.MonthLocator())
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    ax5.set_xlabel('Date', fontsize='large')
    ax5.set_ylabel('Average Compound Sentiment', fontsize='large')
    ax5.set_title('Rating Trends based on Date', fontsize='x-large')
    st.pyplot(fig)

    # --- Chart 7: Feedback Analysis - Sentiment per Feedback Category ---
    st.subheader('Chart 7: Sentiment Analysis for Feedback Categories')
    fig, (ax_chart, ax_text) = plt.subplots(nrows=2, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    sns.barplot(x='review_text', y='compound_sentiment', data=df, ax=ax_chart, palette='viridis')
    ax_chart.set_xlabel('Feedback Category')
    ax_chart.set_ylabel('Average Compound Sentiment')
    ax_chart.set_title('Sentiment Analysis for Feedback Categories')
    ax_chart.tick_params(axis='x', labelrotation=45)
    text_to_display = df['review_text'].iloc[0]
    ax_text.text(0.5, 0.5, text_to_display, ha='center', va='center', fontsize=12, wrap=True)
    ax_text.axis('off')
    st.pyplot(fig)

    # Save the chart to a file or display it
    chart_filename = 'feedback_analysis_chart.png'
    plt.savefig(chart_filename)
    plt.show()

    # Close the PDF file
    pdf_pages.close()

    # Display the PDF path
    st.text(f'Output saved to: {pdf_path}')
