# Setup
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
# Directory Structure
    .
    ├── gym                    # gym environment files
        ├── data                   # folder to store data
        ├── env                   # env setup files
            ├── __init__.py                   # folder init and env registration
            ├── nba-simulation.py                   # env class
            ├── run.py                   # running everything with params passed in here
        ├── utils                   # util functions like data loading
        ├── vis                   # game rendering/visualization functions
        ├── wrapper                   # abstractable wrapping over the env
    ├── models                   # model classes for simulating the game
    ├── README.md                   # README
    └── requirements.txt                   # packages