#!/bin/bash

# Create a new virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate  # For Linux/macOS
env\Scripts\activate    # For Windows
streamlit run rm.py
# Install the required packages
pip install -r nltk.txt
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
/home/adminuser/venv/bin/python -m pip install --upgrade pip
py -m pip install --upgrade nltk
