import streamlit as st
import pandas as pd
import nltk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages
import base64

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
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.hist(df['review_length'], bins=20, color='darkblue')
    ax3.set_xlabel('Review Length', fontsize='large')
    ax3.set_ylabel('Frequency', fontsize='large')
    ax3.set_title('Review Length Analysis', fontsize='x-large')
    st.pyplot(fig3)

    # --- Chart 5: Distribution of Feedback Categories ---
    st.subheader("Distribution of Feedback Categories")
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    sns.countplot(x='feedback_category', data=df, palette='viridis', ax=ax5)
    ax5.set_xlabel('Feedback Category', fontsize='large')
    ax5.set_ylabel('Count', fontsize='large')
    ax5.set_title('Distribution of Feedback Categories', fontsize='x-large')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig5)

# --- Positive and Negative Feedback Categories ---
    st.subheader("Positive and Negative Feedback Categories")

    positive_feedback = df[df['feedback_category'] == 'Positive']['review_text']
    negative_feedback = df[df['feedback_category'] == 'Negative']['review_text']

    st.write("Positive Feedback:")
    for feedback in positive_feedback:
        st.write(feedback)

    st.write("Negative Feedback:")
    for feedback in negative_feedback:
        st.write(feedback)
        
# Function to generate PDF with all charts
def generate_pdf(df):
 # Save all charts to a PDF file
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
        pdf.savefig(fig5)


def main():
    st.title('Sentiment Analysis Dashboard')
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = perform_sentiment_analysis(df)
        render_charts(df)

        # Generate PDF button
        if st.button('Download PDF'):
            pdf_path = generate_pdf(df)
            with open(pdf_path, "rb") as f:
                bytes_data = f.read()
                b64 = base64.b64encode(bytes_data).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="output.pdf">Download PDF file</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
