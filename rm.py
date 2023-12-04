import streamlit as st
import pandas as pd
from PIL import Image
import nltk
from collections import Counter
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud
from dateutil.relativedelta import relativedelta
from PIL import Image, ImageDraw, ImageFont

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
    # --- Chart 1: Sentiment Analysis Based on Ratings ---
    st.subheader("Sentiment Analysis Based On Ratings")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='rating', y='compound_sentiment', data=df, hue='rating', palette='viridis', ax=ax1)
    ax1.set_xticks(list(sorted(df['rating'].unique())))
    ax1.set_xlabel('Rating', fontsize='large')
    ax1.set_ylabel('Compound Sentiment', fontsize='large')
    ax1.set_title('Sentiment Analysis Based On Ratings', fontsize='x-large')
    st.pyplot(fig1)


    # --- Chart 1: Sentiment Analysis Based on Ratings ---
st.subheader("Sentiment Analysis Based On Ratings")
fig, ax = plt.subplots(figsize=(12, 8))

# Assuming 'rating' is categorical and 'compound_sentiment' is numeric
custom_colors = ["#FF5733", "#33FFA8", "#334CFF", "#FF33FF", "#33FFFF"]  # Define your custom colors
sns.barplot(y='rating', x='compound_sentiment', data=df, palette=custom_colors)
ax.set_ylabel('Rating', fontsize='large')
ax.set_xlabel('Compound Sentiment', fontsize='large')
ax.set_title('Sentiment Analysis Based On Ratings', fontsize='x-large')
st.pyplot(fig)



   

    
# --- Positive and Negative Feedback Categories ---
    # ---  st.subheader("Positive and Negative Feedback Categories")
    
    positive_feedback = df[df['feedback_category'] == 'Positive']['review_text']
    negative_feedback = df[df['feedback_category'] == 'Negative']['review_text']
    
    st.write("Positive Feedback:")
    for feedback in positive_feedback:
        st.write(feedback)
    
    st.write("Negative Feedback:")
    for feedback in negative_feedback:
        st.write(feedback)



    # --- Word Frequency Chart ---
    st.subheader("Word Frequency Chart")
    
    # Handle missing values in 'review_text' column
    df['review_text'].fillna('', inplace=True)

    # Combine all review text into a single string
    all_text = ' '.join(df['review_text'])

    # Split the text into words and count their frequencies
    word_counts = Counter(all_text.split())

    # Sort the word counts in descending order
    sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))

    # Plot the top N words by frequency
    N = 20  # Change this to display the top N words
    top_words = list(sorted_word_counts.keys())[:N]
    word_frequencies = [sorted_word_counts[word] for word in top_words]

    plt.figure(figsize=(12, 8))
    plt.barh(top_words, word_frequencies, color='skyblue')
    plt.xlabel('Frequency')
    plt.title('Top {} Word Frequencies'.format(N))
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    st.pyplot(plt)



# --- Chart 2: Average Sentiment Per Business ---
st.subheader("Average Sentiment Per Product OR Service")
fig2, ax2 = plt.subplots(figsize=(12, 8))
colors2 = ['#FFA07A', '#6495ED', '#90EE90']  # Custom colors for each bar
sns.barplot(x='business_column', y='compound_sentiment', data=df, ax=ax2, palette=colors2[:len(df['business_column'].unique())])
ax2.set_xlabel('Business', fontsize='large')
ax2.set_ylabel('Average Compound Sentiment', fontsize='large')
ax2.set_title('Average Sentiment Per Product OR Service', fontsize='x-large')
st.pyplot(fig2)

# --- Chart 3: Review Length Analysis ---
st.subheader("Review Length Analysis")
fig3, ax3 = plt.subplots(figsize=(12, 8))
colors3 = ['#FF6347']  # Custom color for the bar
ax3.hist(df['review_length'], bins=20, color=colors3)
ax3.set_xlabel('Review Length', fontsize='large')
ax3.set_ylabel('Frequency', fontsize='large')
ax3.set_title('Review Length Analysis', fontsize='x-large')
st.pyplot(fig3)

# --- Chart 4: Average Sentiment per Aspect ---
st.subheader("Average Sentiment On Business Aspects")
fig4, ax4 = plt.subplots(figsize=(10, 6))
colors4 = ['#FF4500', '#7B68EE', '#20B2AA']  # Custom colors for each bar
df_filtered['Aspect'] = df_filtered['atmosphere_compound']
avg_sentiment_by_aspect = df_filtered.groupby('Aspect')['compound_sentiment'].mean()
avg_sentiment_by_aspect.plot(kind='bar', color=colors4[:len(avg_sentiment_by_aspect)])
ax4.set_xlabel('Aspect')
ax4.set_ylabel('Average Compound Sentiment')
ax4.set_title('Average Sentiment On Business Aspects')
st.pyplot(fig4)

# --- Chart 7: Sentiment Analysis - Combined Feedback Categories ---
st.subheader("Sentiment Analysis For Feedback Categories")

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




def main():
    st.title('Sentiment Analysis Dashboard')
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = perform_sentiment_analysis(df)
        render_charts(df)

if __name__ == "__main__":
    main()
