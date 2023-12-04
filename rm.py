import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from dateutil.relativedelta import relativedelta

# Download the NLTK VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Define custom colors
custom_colors = {
    'Positive': '#00cc96',
    'Negative': '#EF553B',
    'Suggestion': '#636EFA'
}

def convert_relative_time_to_date(relative_time):
    try:
        if 'months' in relative_time:
            delta = relativedelta(months=int(relative_time.split()[0]))
        else:
            delta = relativedelta(years=int(relative_time.split()[0]))

        return pd.to_datetime('today') - delta
    except ValueError:
        return pd.to_datetime('today')  # Default to today's date for non-numeric values

# Other functions remain the same...

def render_charts(df):
    # --- Chart 1: Sentiment Analysis Based on Ratings ---
    st.subheader("1. Sentiment Analysis Based On Ratings")
    fig1 = px.scatter(df, x='rating', y='compound_sentiment', color='rating', 
                      color_continuous_scale='viridis', labels={'rating': 'Rating', 'compound_sentiment': 'Compound Sentiment'},
                      title='Sentiment Analysis Based On Ratings')
    st.plotly_chart(fig1)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- Chart 2: Average Sentiment Per Business ---
    st.subheader("2. Average Sentiment Per Product OR Service")
    fig2 = px.bar(df, x='business_column', y='compound_sentiment', 
                  labels={'business_column': 'Business', 'compound_sentiment': 'Average Compound Sentiment'},
                  title='Average Sentiment Per Product OR Service')
    st.plotly_chart(fig2)

 # --- Chart 3: Review Length Analysis ---
    st.subheader("3. Review Length Analysis")
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.hist(df['review_length'], bins=20, color='darkblue')
    ax3.set_xlabel('Review Length', fontsize='large')
    ax3.set_ylabel('Frequency', fontsize='large')
    ax3.set_title('Review Length Analysis', fontsize='x-large')
    st.pyplot(fig3)

      # Filter non-null review_text for aspects and their respective sentiment
    df_filtered = df[df['review_text'].notnull()][['atmosphere_compound', 'review_text', 'compound_sentiment']]

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Chart 4: Average Sentiment per Aspect
    st.subheader("4. Average Sentiment On Business Aspects")
    fig, ax = plt.subplots(figsize=(10, 6))
    df_filtered['Aspect'] = df_filtered['atmosphere_compound']
    avg_sentiment_by_aspect = df_filtered.groupby('Aspect')['compound_sentiment'].mean()
    avg_sentiment_by_aspect.plot(kind='bar', color='darkorange', ax=ax)
    ax.set_xlabel('Aspect')
    ax.set_ylabel('Average Compound Sentiment')
    ax.set_title('Average Sentiment On Business Aspects')
    st.pyplot(fig)

    st.markdown("<br><br>", unsafe_allow_html=True)

 # --- Chart 7: Sentiment Analysis - Combined Feedback Categories ---
    st.subheader("5. Sentiment Analysis For Feedback Categories")

    positive_feedback = df[df['feedback_category'] == 'Positive'].sample(n=50, replace=True)
    negative_feedback = df[df['feedback_category'] == 'Negative'].sample(n=50, replace=True)

    combined_feedback = pd.concat([positive_feedback, negative_feedback])
    combined_feedback['feedback_category'] = combined_feedback['feedback_category'].astype(str)

    custom_palette = {'Positive': 'green', 'Negative': 'red'}  # Define your custom colors here

    fig7, ax_chart = plt.subplots(figsize=(12, 8))
    sns.barplot(x='review_text', y='compound_sentiment', hue='feedback_category', 
                data=combined_feedback, palette=custom_palette, ax=ax_chart)
    
    ax_chart.set_xlabel('Review Text', fontsize='large')
    ax_chart.set_ylabel('Average Compound Sentiment', fontsize='large')
    ax_chart.set_title('Sentiment Analysis For Feedback Categories', fontsize='x-large')
    ax_chart.tick_params(axis='x', labelrotation=90)  # Rotate x-axis labels for better readability

    plt.xticks([])  # Remove x-axis text
    plt.tight_layout()

    st.pyplot(fig7)

    st.markdown("<br><br>", unsafe_allow_html=True)

def main():
    st.title('Sentiment Analysis Dashboard')

    # Creating a sidebar for file upload and additional controls
    st.sidebar.title("Options")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = perform_sentiment_analysis(df)

        render_charts(df)

if __name__ == "__main__":
    main()
