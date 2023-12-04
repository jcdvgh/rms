#!/bin/bash

# Create a new virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate  # For Linux/macOS
env\Scripts\activate    # For Windows
streamlit run rm.py
pip install -r requirements.txt
pip install --upgrade streamlit
-m pip install --upgrade pip
pip install nltk --upgrade pip
pip install pandas
pip install numpy
pip install seaborn
pip install matplotlib
pip install wordcloud
pip install python-dateutil
pip install streamlit
py -m pip install --upgrade nltk
pip install click==7.1.2
