# Smart Terrace Farming AI

A Flask-based smart farming dashboard for crop recommendations, soil health monitoring, crop-care suggestions, and (when the disease module and its model files are present) leaf disease detection.

## Features

- Crop recommendations from soil/sensor readings using `crop_model.pkl`
- Soil fertility and soil health scores
- Crop-specific watering, fertilizer, and maintenance advice
- Dashboard served from `templates/dashboard.html`
- Optional advanced monitoring through `advanced_monitoring/`
- Optional leaf disease detection through `disease.py` and its required model files
- Weather data lookup through OpenWeatherMap (requires an API key)

## Project structure

Keep the required files and folders in the project root. Your project should look approximately like this:

```text
project/
├── server.py
├── crop_model.pkl
├── crop_profiles.json          # if used by your version
├── disease.py                  # required for leaf disease features
├── config.py                   # keep if imported by your modules
├── requirements.txt
├── templates/
│   └── dashboard.html
├── advanced_monitoring/        # required for advanced monitoring features
├── models/                     # keep the disease model files required by disease.py
└── README.md
```

The exact contents of `models/`, `advanced_monitoring/`, and any additional configuration/data files depend on your local implementation. Do not delete a file until you have checked whether one of the Python modules imports or loads it.

## Requirements

- Python 3.10 or newer recommended
- pip
- The project files listed above

## Setup on Windows (VS Code)

### 1. Open the project

Open the project folder in VS Code. Open **Terminal → New Terminal** and make sure the terminal is in the same directory as `server.py`.

### 2. Create and activate a virtual environment

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

If `requirements.txt` contains the dependencies for your current project:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the requirements file is incomplete, install the backend's core dependencies:

```bash
pip install Flask Flask-Cors requests joblib pandas numpy scikit-learn
```

If `disease.py` imports additional libraries (for example, a deep-learning framework or image-processing library), install those too. The required packages depend on the actual disease detection implementation and model format.

### 4. Check the model file

`server.py` loads `crop_model.pkl` from the project root. Make sure this file exists beside `server.py`.

The model must be compatible with the scikit-learn/joblib versions used to create it. If you see a model-loading or version error, check which library versions were used when the model was trained.

### 5. Configure the weather API (optional)

In `server.py`, the weather API key is currently a placeholder (`YOUR_API_KEY`). If you want weather data to work, obtain an API key from OpenWeatherMap and configure it without committing secrets to GitHub.

For a quick local test, you can replace the placeholder locally. For a public repository, do **not** commit a real API key. Prefer reading it from an environment variable such as `OPENWEATHER_API_KEY`.

Without a valid key, weather lookup may fail or return the fallback value used by the application.

### 6. Start the server

From the project root, run:

```bash
python server.py
```

The Flask app runs on port `5000`.

Open this address in your browser:

```text
http://127.0.0.1:5000
```

Keep the terminal open while using the dashboard. Press `Ctrl+C` in the terminal to stop the server.

## Hardware / sensor integration

The `/sensor` endpoint accepts JSON sensor readings. The expected fields include:

```json
{
  "N": 80,
  "P": 45,
  "K": 60,
  "temp": 27,
  "moisture": 65,
  "ph": 6.5
}
```

Send the readings as a `POST` request with `Content-Type: application/json` to:

```text
http://127.0.0.1:5000/sensor
```

These are example values for testing, not measurements from a real sensor. The dashboard may wait for sensor data until a reading is received.

## Common issues

### `FileNotFoundError: crop_model.pkl`
Place `crop_model.pkl` in the same directory as `server.py`.

### `TemplateNotFound: dashboard.html`
Make sure the dashboard is located at `templates/dashboard.html`. Flask expects this folder name by default.

### Leaf disease detection does not load
Check that `disease.py`, its imported Python packages, and all model/label files it expects are present. The main server may continue running even if the optional disease module fails to load.

### Advanced monitoring is unavailable
Check that `advanced_monitoring/` contains the modules imported by `server.py`, including `history_manager.py` and `advanced_engine.py`, along with any other files those modules require.

### Port 5000 is already in use
Stop the other process using port 5000, or change the port in the final `app.run(...)` statement in `server.py`.

## GitHub housekeeping

Do not commit local virtual environments, Python caches, private API keys, or large raw datasets unless the project specifically requires them. Keep model files needed to run inference, and check GitHub's file-size limits before committing them.

Suggested `.gitignore` entries:

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
env/
.env
.env.*
!.env.example
.vscode/
.idea/
.DS_Store
Thumbs.db
.ipynb_checkpoints/
.pytest_cache/
*.log
dataset/
tomato_external_dataset/
test_clean/
test_folder/
test_hard/
quick_test_results.csv
```

Review the dataset and test-folder entries before using them: remove any ignore rule for files that are actually required to run the application.

## Notes

- The crop recommendation model is a pre-trained artifact; this repository setup does not retrain it.
- Disease detection and advanced monitoring work only if their own source files, dependencies, and model assets are included.
- Do not upload passwords, tokens, API keys, or other secrets to a public repository.
