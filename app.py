import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ---------------------------------------------------------
# 1. Load Model
# ---------------------------------------------------------
try:
    with open('chatgpt_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature list based on model metadata
FEATURE_NAMES = [
    'topic_category',
    'prompt_style',
    'language',
    'prompt_length',
    'num_examples_in_prompt',
    'clarity_score',
    'specificity_score',
    'token_count',
    'context_window_used_pct',
    'temperature',
    'model_version',
    'response_time_sec',
    'response_length',
    'hallucination_flag',
    'user_rating',
    'follow_up_needed'
]

# ---------------------------------------------------------
# 2. HTML + CSS + JS Template
# ---------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatGPT Analytics & Prediction Hub</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-glow: #6366f1;
            --accent-cyan: #06b6d4;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: rgba(255, 255, 255, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.15) 0px, transparent 50%);
            min-height: 100vh;
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 1000px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group label {
            font-size: 0.825rem;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
            text-transform: capitalize;
        }

        .input-group input {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.9rem;
            outline: none;
            transition: all 0.25s ease;
        }

        .input-group input:focus {
            border-color: var(--accent-glow);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        .actions {
            grid-column: 1 / -1;
            margin-top: 1rem;
            display: flex;
            justify-content: center;
        }

        .btn-submit {
            background: linear-gradient(135deg, var(--accent-glow), var(--accent-cyan));
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.9rem 2.5rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.6);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-panel {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            display: none;
            text-align: center;
            animation: fadeIn 0.4s ease forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-panel h2 {
            font-size: 1.1rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .prediction-badge {
            display: inline-block;
            font-size: 2rem;
            font-weight: 700;
            color: #38bdf8;
            margin-top: 0.25rem;
        }
    </style>
</head>
<body>

    <div class="container">
        <div class="header">
            <h1>ChatGPT Model Predictor</h1>
            <p>Input execution parameters to run real-time inference on your decision tree model</p>
        </div>

        <form id="predictionForm">
            <div class="grid-form">
                <div class="input-group">
                    <label for="topic_category">Topic Category</label>
                    <input type="number" step="any" id="topic_category" name="topic_category" value="1" required>
                </div>
                <div class="input-group">
                    <label for="prompt_style">Prompt Style</label>
                    <input type="number" step="any" id="prompt_style" name="prompt_style" value="1" required>
                </div>
                <div class="input-group">
                    <label for="language">Language</label>
                    <input type="number" step="any" id="language" name="language" value="1" required>
                </div>
                <div class="input-group">
                    <label for="prompt_length">Prompt Length</label>
                    <input type="number" step="any" id="prompt_length" name="prompt_length" value="120" required>
                </div>
                <div class="input-group">
                    <label for="num_examples_in_prompt">Num Examples</label>
                    <input type="number" step="any" id="num_examples_in_prompt" name="num_examples_in_prompt" value="2" required>
                </div>
                <div class="input-group">
                    <label for="clarity_score">Clarity Score</label>
                    <input type="number" step="any" id="clarity_score" name="clarity_score" value="0.85" required>
                </div>
                <div class="input-group">
                    <label for="specificity_score">Specificity Score</label>
                    <input type="number" step="any" id="specificity_score" name="specificity_score" value="0.90" required>
                </div>
                <div class="input-group">
                    <label for="token_count">Token Count</label>
                    <input type="number" step="any" id="token_count" name="token_count" value="450" required>
                </div>
                <div class="input-group">
                    <label for="context_window_used_pct">Context Window Used (%)</label>
                    <input type="number" step="any" id="context_window_used_pct" name="context_window_used_pct" value="25.5" required>
                </div>
                <div class="input-group">
                    <label for="temperature">Temperature</label>
                    <input type="number" step="any" id="temperature" name="temperature" value="0.7" required>
                </div>
                <div class="input-group">
                    <label for="model_version">Model Version</label>
                    <input type="number" step="any" id="model_version" name="model_version" value="4" required>
                </div>
                <div class="input-group">
                    <label for="response_time_sec">Response Time (sec)</label>
                    <input type="number" step="any" id="response_time_sec" name="response_time_sec" value="1.2" required>
                </div>
                <div class="input-group">
                    <label for="response_length">Response Length</label>
                    <input type="number" step="any" id="response_length" name="response_length" value="600" required>
                </div>
                <div class="input-group">
                    <label for="hallucination_flag">Hallucination Flag (0/1)</label>
                    <input type="number" step="any" id="hallucination_flag" name="hallucination_flag" value="0" required>
                </div>
                <div class="input-group">
                    <label for="user_rating">User Rating</label>
                    <input type="number" step="any" id="user_rating" name="user_rating" value="5" required>
                </div>
                <div class="input-group">
                    <label for="follow_up_needed">Follow Up Needed (0/1)</label>
                    <input type="number" step="any" id="follow_up_needed" name="follow_up_needed" value="0" required>
                </div>

                <div class="actions">
                    <button type="submit" class="btn-submit">Run Prediction</button>
                </div>
            </div>
        </form>

        <div id="resultPanel" class="result-panel">
            <h2>Predicted Output Class</h2>
            <div id="predictionOutput" class="prediction-badge">-</div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            const resultPanel = document.getElementById('resultPanel');
            const output = document.getElementById('predictionOutput');

            if (result.success) {
                output.innerText = `Class: ${result.prediction}`;
                resultPanel.style.display = 'block';
            } else {
                output.innerText = `Error: ${result.error}`;
                resultPanel.style.display = 'block';
            }
        });
    </script>
</body>
</html>
"""

# ---------------------------------------------------------
# 3. Flask Routes
# ---------------------------------------------------------
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load on server.'}), 500

    try:
        input_data = []
        for feature in FEATURE_NAMES:
            val = request.request.form.get(feature, 0) if hasattr(request, 'request') else request.form.get(feature, 0)
            input_data.append(float(val))

        features_array = np.array([input_data])
        prediction = model.predict(features_array)[0]

        return jsonify({
            'success': True,
            'prediction': int(prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ---------------------------------------------------------
# 4. App Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
