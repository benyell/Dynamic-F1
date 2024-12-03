# SHY-but-not-for-too-long
Supply Chain Optimization in F1 races using Reinforcement learning

First clone this repository onto Visual code

Second, run this command: pip install -r requirements.txt
This will install all the dependencies onto your computer

A substep after this is that you need to create a venv
if not created: type this in bash: .\venv\Scripts\activate

I have already created the necesary files for the usage of a venv. use bach t

Run this command only if your vscode did not notify about a new environment : python -m venv venv

Third, run this command: python src/data_collection.py 
This will collect data from fastF1 repo

Fourth, run this command: python src/generate_dataset.py
then, run this command: python src/generate_synthetic_data.py

Fifth, we need to run python data_preprocessing.py

Sixth, run python src/train_rl_agent.py
Then, run this command: python src/test_rl_agent.py

Last, run streamlit run src/app.py

This will run the front end application with all the graphs.