import os
from flask import Flask, render_template, request
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import sys 
from src.pipelines.prediction_pipeline import PredictPipeline  # Update with your actual import path for PredictPipeline

app = Flask(__name__, static_url_path='/static')

# Path to combined_data.csv
DATA_FILE = 'notebooks/data/Prediction_data.csv'  # Update with your actual file path

players = {
    "MI": ["Rohit Sharma", "Suryakumar Yadav", "Ishan Kishan", "Hardik Pandya", "Krunal Pandya", "Kieron Pollard", "Trent Boult", "Jasprit Bumrah", "Rahul Chahar", "Quinton de Kock", "Nathan Coulter-Nile"],
    "CSK": ["MS Dhoni", "Suresh Raina", "Ravindra Jadeja", "Faf du Plessis", "Deepak Chahar", "Dwayne Bravo", "Sam Curran", "Ruturaj Gaikwad", "Ambati Rayudu", "Shardul Thakur", "Imran Tahir"],
    "RCB": ["Virat Kohli", "AB de Villiers", "Glenn Maxwell", "Yuzvendra Chahal", "Harshal Patel", "Devdutt Padikkal", "Mohammed Siraj", "Washington Sundar", "Navdeep Saini", "Kyle Jamieson", "Dan Christian"],
    "KKR": ["Eoin Morgan", "Andre Russell", "Dinesh Karthik", "Pat Cummins", "Shubman Gill", "Varun Chakravarthy", "Lockie Ferguson", "Nitish Rana", "Sunil Narine", "Kuldeep Yadav", "Shivam Mavi"],
    "SRH": ["David Warner", "Kane Williamson", "Rashid Khan", "Bhuvneshwar Kumar", "Manish Pandey", "Jonny Bairstow", "Jason Holder", "T Natarajan", "Vijay Shankar", "Sandeep Sharma", "Wriddhiman Saha"],
    "DC": ["Rishabh Pant", "Shikhar Dhawan", "Prithvi Shaw", "Kagiso Rabada", "Marcus Stoinis", "Anrich Nortje", "Shimron Hetmyer", "Ravichandran Ashwin", "Avesh Khan", "Ajinkya Rahane", "Chris Woakes"],
    "PBKS": ["KL Rahul", "Chris Gayle", "Mayank Agarwal", "Mohammed Shami", "Nicholas Pooran", "Ravi Bishnoi", "Jhye Richardson", "Shahrukh Khan", "Deepak Hooda", "Arshdeep Singh", "Mandeep Singh"],
    "RR": ["Sanju Samson", "Jos Buttler", "Ben Stokes", "Jofra Archer", "Chris Morris", "Rahul Tewatia", "Riyan Parag", "Shreyas Gopal", "Mustafizur Rahman", "Yashasvi Jaiswal", "David Miller"],
    "GT": ["Shubman Gill", "Rashid Khan", "David Miller", "Hardik Pandya", "Rahul Tewatia", "Matthew Wade", "Lockie Ferguson", "Mohammed Shami", "Alzarri Joseph", "Yash Dayal", "Vijay Shankar"],
    "LSG": ["KL Rahul", "Quinton de Kock", "Deepak Hooda", "Marcus Stoinis", "Jason Holder", "Krunal Pandya", "Mark Wood", "Avesh Khan", "Ravi Bishnoi", "Manish Pandey", "Dushmantha Chameera"]
}

columns_needed = [
    'fullName', 'batting_position', 'runs', 'balls', 'fours', 'sixes', 'strike_rate',
    '50_runs', '100_runs', '30_runs', 'Duck', 'overs', 'dots', 'maidens', 'conceded',
    'foursConceded', 'sixesConceded', 'wickets', 'economyRate', 'wides', 'noballs',
    'LBW', 'Hitwicket', 'CaughtBowled', 'Bowled', '3_wickets',
    '4_wickets', '5_wickets', 'ecoPoints', 'catching_FP', 'stumping_FP',
    'direct_runout_FP', 'indirect_runout_FP', 'Starting_11', 'Captain', 'Vice Captain'
    
]

# Function to fetch recent match data for a player from combined_data.csv
def fetch_player_data(all_players, team1, team2):
    try:
        # Read combined_data.csv into DataFrame
        combined_data = pd.read_csv(DATA_FILE)
        
        # Filter data for the players listed in all_players
        filtered_df = combined_data[combined_data['fullName'].isin(all_players)]

        # Define the conditions to filter matches between team1 and team2
        team1_vs_team2_condition = (
            ((filtered_df['home_team'] == team1) & (filtered_df['away_team'] == team2)) |
            ((filtered_df['home_team'] == team2) & (filtered_df['away_team'] == team1))
        )

        # Get the data for the last 3 matches (excluding matches between team1 and team2)
        past_matches_df = filtered_df[~team1_vs_team2_condition].groupby('fullName').tail(3)

        # Get the data for the last 2 matches between team1 and team2
        team1_vs_team2_matches_df = filtered_df[team1_vs_team2_condition].groupby('fullName').tail(2)

        # Combine the data
        combined_test = pd.concat([past_matches_df, team1_vs_team2_matches_df])
        grouped_data = combined_test[columns_needed]
        grouped_data = grouped_data.groupby('fullName').mean().reset_index()

        # Return DataFrame with player's match data
        return grouped_data

    except Exception as e:
        logging.error(f'Exception occurred while fetching data for {all_players}: {e}')
        return pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit_teams', methods=['POST'])
def submit_teams():
    team1 = request.form['team1']
    team2 = request.form['team2']
    
    team1_players = players.get(team1, [])
    team2_players = players.get(team2, [])
    
    combined_players = team1_players + team2_players
    prediction_data = fetch_player_data(combined_players, team1, team2)

    try:
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        model_path = os.path.join('artifacts', 'model.pkl')

        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)

        data_scaled = preprocessor.transform(prediction_data.select_dtypes(include='number'))

        pred = model.predict(data_scaled)

        # Add predictions to DataFrame
        prediction_data['Total_FP'] = pred

        # Sort players based on predictions or other criteria
        prediction_data.sort_values(by='Total_FP', ascending=False, inplace=True)

        # Get top 11 players combined from both teams
        top_players = prediction_data.head(11)['fullName'].tolist()

        # Render result.html with top players
        return render_template('result.html', team1=team1, team2=team2, top_players=top_players)
    
    except Exception as e:
        logging.error("Exception occurred in prediction")
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(debug=True)
