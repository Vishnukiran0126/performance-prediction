import time
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# Function to scrape player stats dynamically from the web
def scrape_player_data(player_name):
    driver = webdriver.Chrome()
    #search_url = f"https://stats.espncricinfo.com/ci/engine/player/253802.html?class=2;filter=advanced;floodlit=1;floodlit=2;search_player={player_name};template=results;type=batting;view=match"
    #driver.get(search_url)
    # Open the target page
    driver.get("https://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;type=batting")

    # Wait for the page to load (adjust the time if necessary)
    time.sleep(3)  # Wait 3 seconds for the page to load

    # Find the player name input field by XPath
    player_name_input = driver.find_element(By.XPATH, "/html/body/div[3]/div[1]/div[1]/div[3]/div[3]/form/input[1]")

    # Enter the player's name (replace with the name given by the user)
    player_name_input.send_keys(player_name)  # You can replace 'Virat Kohli' with the user's input

    # Find the submit button by its class name or XPath
    submit_button = driver.find_element(By.XPATH, "/html/body/div[3]/div[1]/div[1]/div[3]/div[3]/form/input[2]")

    # Click the submit button to search for the player
    submit_button.click()

    # Wait for the results to load (adjust the time if necessary)
    time.sleep(5)  # Adjust the time depending on the page load time
    linkbtn=driver.find_element(By.XPATH,"/html/body/div[3]/div[1]/div[1]/div[3]/div/div[2]/ul/li[1]/a")
    linkbtn.click()

    odilink=driver.find_element(By.XPATH,"/html/body/div[3]/div[1]/div[1]/div[3]/div/div[4]/table/tbody/tr/td[3]/a[2]")
    # Optionally, you can check the result or print out the page title after submission
    odilink.click()
    #time.sleep(4)

    batrec=driver.find_element(By.XPATH,"/html/body/div[3]/div[1]/div[1]/div[3]/div/table[1]/tbody/tr/td/table/tbody/tr[2]/td/form/table/tbody/tr[10]/td[2]/table/tbody/tr[2]/td/table/tbody/tr/td[2]/label/input")
    batrec.click()

    radiobtn=driver.find_element(By.XPATH,"/html/body/div[3]/div[1]/div[1]/div[3]/div/table[1]/tbody/tr/td/table/tbody/tr[2]/td/form/table/tbody/tr[10]/td[2]/table/tbody/tr[3]/td[1]/label[3]/input")
    radiobtn.click()

    subquery=driver.find_element(By.XPATH,"/html/body/div[3]/div[1]/div[1]/div[3]/div/table[1]/tbody/tr/td/table/tbody/tr[2]/td/form/table/tbody/tr[11]/td[2]/table/tbody/tr/td[1]/input")
    subquery.click()
    print(driver.title) 

    # Find the table using XPath
    table = driver.find_element(By.XPATH, "/html/body/div[3]/div/div[1]/div[3]/table[4]/tbody")
    #/html/body/div[3]/div/div[1]/div[3]/table[4]/tbody

    # Extract all rows in the table
    rows = table.find_elements(By.XPATH, ".//tr")

    # Initialize an empty list to store the extracted data
    data = []

    # Loop through the rows and extract column values
    for row in rows:
        columns = row.find_elements(By.XPATH, ".//td")
        if columns:
            # Extract and clean each column text
            data.append([col.text.strip() for col in columns])

    # Close the driver
    driver.quit()

    # Log the number of rows extracted
    print(f"Number of rows extracted from the webpage: {len(data)}")

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        'Runs', 'Balls Faced', 'Batting Score', 'Strike Rate', 'Fours', 'Sixes',
        'Empty', 'Opposition', 'Ground', 'Start Date', 'Match ID'
    ])

    # Remove the empty column
    df = df.drop(columns=['Empty'])
    original_row_count = df.shape[0]
    # Drop rows where the 'Runs' column has the value 'DNB' (Did Not Bat)
    df = df[df['Runs'] != 'DNB']

    df = df[df['Runs'] != 'TDNB']
    print(f"Number of rows after removing 'DNB' and 'TDNB': {df.shape[0]} (Cleaned {original_row_count - df.shape[0]} rows)")


    # Replace '-' with NaN in the DataFrame
    df.replace('-', np.nan, inplace=True)

    # Check for columns with missing values
    for column in df.columns:
        if df[column].isnull().any():  # Check for NaN values
            print(f"Column '{column}' contains missing values")
    
    

    # Convert relevant columns to numeric and replace invalid values with NaN
    df['Runs'] = pd.to_numeric(df['Runs'], errors='coerce')
    df['Balls Faced'] = pd.to_numeric(df['Balls Faced'], errors='coerce')
    df['Strike Rate'] = pd.to_numeric(df['Strike Rate'], errors='coerce')
    df['Fours'] = pd.to_numeric(df['Fours'], errors='coerce')
    df['Sixes'] = pd.to_numeric(df['Sixes'], errors='coerce')

    # Impute missing values in numeric columns with the median of each column
    df['Runs'].fillna(df['Runs'].median(), inplace=True)
    df['Balls Faced'].fillna(df['Balls Faced'].median(), inplace=True)
    df['Strike Rate'].fillna(df['Strike Rate'].median(), inplace=True)
    df['Fours'].fillna(df['Fours'].median(), inplace=True)
    df['Sixes'].fillna(df['Sixes'].median(), inplace=True)

    # Impute missing categorical columns with the mode (most frequent value)
    df['Opposition'].fillna(df['Opposition'].mode()[0], inplace=True)
    df['Ground'].fillna(df['Ground'].mode()[0], inplace=True)

    # Extract date features (e.g., days from the last match)
    df['Start Date'] = pd.to_datetime(df['Start Date'], format='%d %b %Y')
    df['Days Since Last Match'] = (df['Start Date'] - df['Start Date'].max()).dt.days

    # Encode categorical features (e.g., Opposition, Ground)
    le = LabelEncoder()
    df['Opposition'] = le.fit_transform(df['Opposition'])
    df['Ground'] = le.fit_transform(df['Ground'])

    # Feature selection: drop the 'Match ID' and 'Start Date' as they are not useful for prediction
    X = df.drop(columns=['Match ID', 'Start Date'])

    # Predict both 'Runs' and 'Balls Faced'
    y = df[['Runs', 'Balls Faced']]  # Now predicting two outputs

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data (optional but often helpful for regression tasks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build a Random Forest model (you can experiment with other models like Linear Regression, LSTM, etc.)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict the score for the next match (predict for the first test set example)
    predicted_values = model.predict([X_test[0]])  # Predict for the first test set example
    predicted_runs = predicted_values[0][0]
    predicted_balls_faced = predicted_values[0][1]

    print(f"Predicted Runs: {predicted_runs}")
    print(f"Predicted Balls Faced: {predicted_balls_faced}")

    # Evaluate the model's performance (efficiency and accuracy)
    y_pred = model.predict(X_test)

    # Calculate MAE, MSE, and R2 for both 'Runs' and 'Balls Faced'
    mae_runs = mean_absolute_error(y_test['Runs'], y_pred[:, 0])
    mae_balls = mean_absolute_error(y_test['Balls Faced'], y_pred[:, 1])

    mse_runs = mean_squared_error(y_test['Runs'], y_pred[:, 0])
    mse_balls = mean_squared_error(y_test['Balls Faced'], y_pred[:, 1])

    r2_runs = r2_score(y_test['Runs'], y_pred[:, 0])
    r2_balls = r2_score(y_test['Balls Faced'], y_pred[:, 1])

    # Print evaluation metrics
    print(f"MAE (Runs): {mae_runs}")
    print(f"MAE (Balls Faced): {mae_balls}")
    print(f"MSE (Runs): {mse_runs}")
    print(f"MSE (Balls Faced): {mse_balls}")
    print(f"R² (Runs): {r2_runs}")
    print(f"R² (Balls Faced): {r2_balls}")
    return int(round(predicted_runs,0)),int(round(predicted_balls_faced,0))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    player_name = request.json.get("player_name")

    try:
    
        # Predict the player's performance
        predicted_runs, predicted_balls_faced = scrape_player_data(player_name)
        print(predicted_runs)
        print(predicted_balls_faced)

        return jsonify({
            "player_name": player_name,
            "predicted_runs": int(predicted_runs),
            "predicted_balls_faced": int(predicted_balls_faced)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
