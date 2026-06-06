# Notebooks

Run the notebooks **in order** from 00 to 05.  
Each notebook is self-contained and references the production code in `src/`.

| Notebook | What it covers |
|---|---|
| `00_business_understanding` | Problem framing, stakeholder context, success criteria |
| `01_data_understanding` | Load & profile data, time-based train/validation split |
| `02_exploratory_analysis` | Business hypotheses, correlations, seasonality plots |
| `03_feature_engineering` | `RossmannFeatureTransformer`, cyclical encoding, pipeline |
| `04_modeling_and_business_results` | Model comparison, error analysis, financial scenarios |
| `05_deployment_and_consumption` | API, Streamlit, Telegram bot demo |

## Requirements

All notebooks depend on the main project package.  
Install everything with:

```bash
make setup        # from the repo root
```

## Google Colab

You can run the notebooks on Google Colab without a local environment:

```python
# Cell 1 — clone and install
!git clone https://github.com/<your-username>/rossmann-store-sales.git project
%cd project
!pip install -q -r requirements.txt

# Cell 2 — download Kaggle data
from google.colab import files
files.upload()          # upload kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!pip install -q kaggle
!kaggle competitions download -c rossmann-store-sales -p data/raw
!unzip -q -o data/raw/rossmann-store-sales.zip -d data/raw

# Cell 3 — train
!PYTHONPATH=src python -m rossmann_store_sales.cli train
```
