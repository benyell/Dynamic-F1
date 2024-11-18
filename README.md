# SHY-but-not-for-too-long
Supply Chain Optimization in F1 races using Reinforcement learning

First clone this repository onto Visual code

Second, run this command: pip install -r requirements.txt
This will install all the dependencies onto your computer

A substep after this is that you need to create a venv
I have already created the necesary files for the usage of a venv.

Run this command only if your vscode did not notify about a new environment : python -m venv venv

Third, run this command: python data_collection.py 
THis will collect data from fastF1 repo

Fourth, run this command: python generate_synthetic_data.py
This will generate synthetic demand surges. We still have to make more effort here.

Fifth, run this command: python data_preprocessing.py
This will preprocess the data and give us a plot of data avaialable with us.

By this point, we can understand the the Data gathering and preprocessing phase. 