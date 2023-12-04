import streamlit as st
import pandas as pd
import nltk
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud
from dateutil.relativedelta import relativedelta

# Download the NLTK VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

def convert_relative_time_to_date(relative_time):
    try:
        if 'months' in relative_time:
            delta = relativedelta(months=int(relative_time.split()[0]))
        else:
            delta = relativedelta(years=int(relative_time.split()[0]))

        return pd.to_datetime('today') - delta
    except ValueError:
        return pd.to_datetime('today')  # Default to today's date for non-numeric values

def perform_sentiment_analysis(df):
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

    return df

def render_charts(df):
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # --- Chart 1: Sentiment Analysis based on Ratings ---
    st.subheader("Sentiment Analysis based on Ratings")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='rating', y='compound_sentiment', data=df, hue='rating', palette='viridis', ax=ax1)
    ax1.set_xticks(list(sorted(df['rating'].unique())))
    ax1.set_xlabel('Rating', fontsize='large')
    ax1.set_ylabel('Compound Sentiment', fontsize='large')
    ax1.set_title('Sentiment Analysis based on Ratings', fontsize='x-large')
    st.pyplot(fig1)

    # --- Chart 2: Average Sentiment per Business ---
    st.subheader("Average Sentiment per Business")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.barplot(x='business_column', y='compound_sentiment', data=df, ax=ax2, palette='viridis')
    ax2.set_xlabel('Business', fontsize='large')
    ax2.set_ylabel('Average Compound Sentiment', fontsize='large')
    ax2.set_title('Average Sentiment Per Product OR Service', fontsize='x-large')
    st.pyplot(fig2)

    # --- Chart 3: Review Length Analysis ---
    st.subheader("Review Length Analysis")
    pd.set_option('mode.use_inf_as_null', True)  # Set the pandas option directly
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.histplot(df['review_length'], bins=20, color='darkblue', kde=False, ax=ax3)
    ax3.set_xlabel('Review Length', fontsize='large')
    ax3.set_ylabel('Frequency', fontsize='large')
    ax3.set_title('Review Length Analysis', fontsize='x-large')
    st.pyplot(fig3)

    # --- Chart 4: Word Cloud - Frequency of Mentions ---
    st.subheader("Word Cloud - Frequency of Mentions")
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white',
                          colormap='viridis').generate(' '.join(df['review_text']))
    ax4.imshow(wordcloud, interpolation='bilinear')
    ax4.axis('off')
    ax4.set_title('The Most Visible and Popular Words', fontsize='x-large')
    st.pyplot(fig4)

    # --- Chart 5: Feedback Analysis - Distribution of Feedback Categories ---
    st.subheader("Feedback Analysis - Distribution of Feedback Categories")
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    sns.countplot(x='feedback_category', data=df, palette='viridis', ax=ax5)
    ax5.set_xlabel('Feedback Category', fontsize='large')
    ax5.set_ylabel('Count', fontsize='large')
    ax5.set_title('Distribution of Feedback Categories', fontsize='x-large')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig5)

    # --- Chart 6: Sentiment Trends based on Time Periods ---
    st.subheader("Sentiment Trends based on Time Periods")
    pd.set_option('mode.use_inf_as_null', True)  # Set the pandas option directly
    fig6, ax6 = plt.subplots(figsize=(12, 8))
    sns.lineplot(x='converted_date', y='compound_sentiment', data=df, ax=ax6)
    ax6.xaxis.set_major_locator(mdates.MonthLocator())
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    ax6.set_xlabel('Date', fontsize='large')
    ax6.set_ylabel('Average Compound Sentiment', fontsize='large')
    ax6.set_title('Sentiment Trends based on Time Periods', fontsize='x-large')
    st.pyplot(fig6)

    # --- Chart 7: Feedback Analysis - Sentiment per Feedback Category ---
    st.subheader("Feedback Analysis - Sentiment per Feedback Category")
    fig7, (ax_chart, ax_text) = plt.subplots(nrows=2, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    sns.barplot(x='review_text', y='compound_sentiment', data=df, ax=ax_chart, palette='viridis')
    ax_chart.set_xlabel('Feedback Category')
    ax_chart.set_ylabel('Average Compound Sentiment')
    ax_chart.set_title('Sentiment Analysis for Feedback Categories')
    ax_chart.tick_params(axis='x', labelrotation=45)
    text_to_display = df['review_text'].iloc[0]
    ax_text.text(0.5, 0.5, text_to_display, ha='center', va='center', fontsize=12, wrap=True)
    ax_text.axis('off')
    chart_filename = 'feedback_analysis_chart.png'
    plt.savefig(chart_filename)
    st.pyplot(fig7)
    
def main():
    st.title('Sentiment Analysis Dashboard')
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = perform_sentiment_analysis(df)
        render_charts(df)

if __name__ == "__main__":
    main()
