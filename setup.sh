#!/bin/bash

# Create a new virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate  # For Linux/macOS
env\Scripts\activate    # For Windows

# Install the required packages
pip install -r requirements.txt
pip install --upgrade streamlit
-m pip install --upgrade pip
pip install streamlit
pip install requests
pip install bs4
pip install BeautifulSoup
pip install pandas
pip install pd
pip install time
pip install nltk
pip install numpy
pip install np
pip install nltk.sentiment 
pip install SentimentIntensityAnalyzer
pip install seaborn
pip install sns
pip install matplotlib.pyplot 
pip install plt
pip install matplotlib.gridspec 
pip install gridspec
pip install matplotlib.dates 
pip install mdates
pip install matplotlib.backends.backend_pdf 
pip install PdfPages
pip install wordcloud 
pip install WordCloud
pip install dateutil.relativedelta 
pip install relativedelta
