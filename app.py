import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# Base directory for Vercel Serverless environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'chatgpt_model.pkl')

app = Flask(__name__)

# Safe model load
model = None
model_error = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        model_error = f"Model loading error: {str(e)}"
else:
    model_error = f"Model file not found at: {MODEL_PATH}"

# Mappings from Name Options -> Numerical Model Encodings
TOPIC_MAP = {"Coding": 0, "Creative Writing": 1, "Data Analysis": 2, "General Knowledge": 3, "Math & Logic": 4}
PROMPT_STYLE_MAP = {"Direct": 0, "Roleplay": 1, "Step-by-Step": 2, "Few-Shot": 3}
LANGUAGE_MAP = {"English": 0, "Spanish": 1, "French": 2, "German": 3, "Hindi": 4}
MODEL_VERSION_MAP = {"GPT-3.5-Turbo": 0, "GPT-4": 1, "GPT-4o": 2, "GPT-4-Mini": 3}
FLAG_MAP = {"No": 0, "Yes": 1}

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatGPT Quality Evaluator — AI Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #172554 100%);
            --card-bg: rgba(255, 255, 255, 0.03);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2.5rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 1000px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header h1 {
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(to right, #a78bfa, #38bdf8, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #cbd5e1;
            letter-spacing: 0.025em;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            color: #fff;
            font-size: 0.9rem;
            outline: none;
            transition: all 0.25s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent-purple), #6d28d9);
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -5px rgba(109, 40, 217, 0.4);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--card-border);
            display: none;
            text-align: center;
        }

        .result-box.active {
            display: block;
            animation: fadeIn 0.4s ease-out;
        }

        .result-title {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }

        .result-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-cyan);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>ChatGPT Quality Predictor</h1>
        <p>Select prompt and execution metrics using human-readable options</p>
    </div>

    <form id="chatgptForm">
        <div class="form-grid">
            <div class="input-group">
                <label>Topic Category</label>
                <select name="topic_category" required>
                    <option value="Coding">Coding</option>
                    <option value="Creative Writing">Creative Writing</option>
                    <option value="Data Analysis">Data Analysis</option>
                    <option value="General Knowledge">General Knowledge</option>
                    <option value="Math & Logic">Math & Logic</option>
                </select>
            </div>

            <div class="input-group">
                <label>Prompt Style</label>
                <select name="prompt_style" required>
                    <option value="Direct">Direct</option>
                    <option value="Roleplay">Roleplay</option>
                    <option value="Step-by-Step">Step-by-Step</option>
                    <option value="Few-Shot">Few-Shot</option>
                </select>
            </div>

            <div class="input-group">
                <label>Language</label>
                <select name="language" required>
                    <option value="English">English</option>
                    <option value="Spanish">Spanish</option>
                    <option value="French">French</option>
                    <option value="German">German</option>
                    <option value="Hindi">Hindi</option>
                </select>
            </div>

            <div class="input-group">
                <label>Prompt Length (Chars)</label>
                <input type="number" name="prompt_length" value="150" required>
            </div>

            <div class="input-group">
                <label>Examples in Prompt</label>
                <input type="number" name="num_examples_in_prompt" value="2" min="0" max="10" required>
            </div>

            <div class="input-group">
                <label>Clarity Score (1-10)</label>
                <input type="number" step="0.1" name="clarity_score" value="8.5" min="1" max="10" required>
            </div>

            <div class="input-group">
                <label>Specificity Score (1-10)</label>
                <input type="number" step="0.1" name="specificity_score" value="7.8" min="1" max="10" required>
            </div>

            <div class="input-group">
                <label>Token Count</label>
                <input type="number" name="token_count" value="450" required>
            </div>

            <div class="input-group">
                <label>Context Window Used (%)</label>
                <input type="number" step="0.1" name="context_window_used_pct" value="25.0" min="0" max="100" required>
            </div>

            <div class="input-group">
                <label>Temperature (0.0 - 1.0)</label>
                <input type="number" step="0.1" name="temperature" value="0.7" min="0" max="1" required>
            </div>

            <div class="input-group">
                <label>Model Version</label>
                <select name="model_version" required>
                    <option value="GPT-4o">GPT-4o</option>
                    <option value="GPT-4">GPT-4</option>
                    <option value="GPT-4-Mini">GPT-4-Mini</option>
                    <option value="GPT-3.5-Turbo">GPT-3.5-Turbo</option>
                </select>
            </div>

            <div class="input-group">
                <label>Response Time (Sec)</label>
                <input type="number" step="0.1" name="response_time_sec" value="2.3" required>
            </div>

            <div class="input-group">
                <label>Response Length (Words)</label>
                <input type="number" name="response_length" value="350" required>
            </div>

            <div class="input-group">
                <label>Hallucination Flag</label>
                <select name="hallucination_flag" required>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>

            <div class="input-group">
                <label>User Rating (1-5)</label>
                <input type="number" step="0.1" name="user_rating" value="4.5" min="1" max="5" required>
            </div>

            <div class="input-group">
                <label>Follow-up Needed</label>
                <select name="follow_up_needed" required>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>

            <button type="submit" class="btn-submit">Evaluate ChatGPT Performance</button>
        </div>
    </form>

    <div id="resultBox" class="result-box">
        <div class="result-title">Model Classification Result</div>
        <div id="resultValue" class="result-value">--</div>
    </div>
</div>

<script>
    document.getElementById('chatgptForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const resultBox = document.getElementById('resultBox');
        const resultValue = document.getElementById('resultValue');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                resultBox.classList.add('active');
                resultValue.textContent = String(data.prediction).toUpperCase();
            } else {
                alert('Prediction Error: ' + data.error);
            }
        } catch (err) {
            alert('Server connection error!');
        }
    });
</script>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_LAYOUT)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': model_error or 'Model file unavailable.'}), 500

    try:
        data = request.form if request.form else (request.get_json(silent=True) or {})

        # Map named choices back to numerical model encodings safely
        topic_val = TOPIC_MAP.get(data.get('topic_category'), 0)
        style_val = PROMPT_STYLE_MAP.get(data.get('prompt_style'), 0)
        lang_val = LANGUAGE_MAP.get(data.get('language'), 0)
        prompt_len = float(data.get('prompt_length', 150))
        num_examples = float(data.get('num_examples_in_prompt', 2))
        clarity = float(data.get('clarity_score', 8.5))
        specificity = float(data.get('specificity_score', 7.8))
        tokens = float(data.get('token_count', 450))
        context_used = float(data.get('context_window_used_pct', 25.0))
        temp = float(data.get('temperature', 0.7))
        model_ver = MODEL_VERSION_MAP.get(data.get('model_version'), 0)
        resp_time = float(data.get('response_time_sec', 2.3))
        resp_len = float(data.get('response_length', 350))
        hallucination = FLAG_MAP.get(data.get('hallucination_flag'), 0)
        rating = float(data.get('user_rating', 4.5))
        follow_up = FLAG_MAP.get(data.get('follow_up_needed'), 0)

        # 16 ordered features expected by chatgpt_model.pkl
        features = np.array([[
            topic_val, style_val, lang_val, prompt_len, num_examples,
            clarity, specificity, tokens, context_used, temp,
            model_ver, resp_time, resp_len, hallucination, rating, follow_up
        ]])

        prediction = model.predict(features)[0]

        return jsonify({
            'success': True,
            'prediction': str(prediction)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# WSGI Handler export for Vercel
app = app

if __name__ == '__main__':
    app.run(debug=True)
