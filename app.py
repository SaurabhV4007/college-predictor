from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('college_predictor_model_v2.pkl')
encoders = {
    'Category': joblib.load('Category_encoder.pkl'),
    'Branch_Name': joblib.load('Branch_Name_encoder.pkl'),
    'College_Name': joblib.load('College_Name_encoder.pkl'),
    'Location': joblib.load('Location_encoder.pkl')
}

# Load original data
data = pd.read_csv('College_Category_Score_Summary.csv', encoding="windows-1252", on_bad_lines='skip')

# Preprocess and transform
def prepare_data():
    score_columns = {
        'OPEN': 'OPEN_Score', 'OBC': 'OBC_Score', 'SC': 'SC_Score',
        'ST': 'ST_Score', 'SBC': 'SBC_Score', 'DT/VJ': 'DT/VJ_Score'
    }
    dfs = []
    for category, col in score_columns.items():
        temp = data[['College_Name', 'Branch_Name', 'Location', col]].copy()
        temp = temp.rename(columns={col: 'MHT_CET_Score'})
        temp['Category'] = category
        temp = temp.dropna(subset=['MHT_CET_Score'])
        dfs.append(temp)
    df = pd.concat(dfs, ignore_index=True)
    return df

df = prepare_data()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.get_json()
        category = user_input['Category']
        score = float(user_input['MHT_CET_Score'])
        branch = user_input['Branch_Name']
        location = user_input['Location']

        input_data = {
            'Category': encoders['Category'].transform([category])[0],
            'MHT_CET_Score': score,
            'Branch_Name': encoders['Branch_Name'].transform([branch])[0],
            'Location': encoders['Location'].transform([location])[0] if location != "All" else -1
        }

        input_df = pd.DataFrame([input_data])
        probabilities = model.predict_proba(input_df)[0]
        all_colleges = encoders['College_Name'].inverse_transform(np.arange(len(probabilities)))

        result_df = pd.DataFrame({
            'College_Name': all_colleges,
            'Probability': probabilities
        })

        # Decode original data for merging
        df_decoded = df.copy()
        df_decoded['College_Name'] = df_decoded['College_Name'].str.strip()
        unique_cols = df_decoded[['College_Name', 'Branch_Name', 'Location']].drop_duplicates()

        result_df = result_df.merge(unique_cols, on='College_Name', how='left')

        # Filter
        filtered = result_df[result_df['Branch_Name'] == branch]
        if location != "All":
            filtered = filtered[filtered['Location'] == location]

        if filtered.empty:
            return jsonify([])

        # Normalize and format
        filtered['Probability'] = filtered['Probability'] / filtered['Probability'].sum()
        filtered = filtered.sort_values(by='Probability', ascending=False).head(15)
        filtered['Probability'] = 80 + (filtered['Probability'] * 19)

        results = filtered[['College_Name', 'Branch_Name', 'Location', 'Probability']].round(2).to_dict(orient='records')
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

