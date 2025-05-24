![Contributors](https://img.shields.io/badge/contributor-Thilini-mediumseagreen)![Contributors](https://img.shields.io/badge/contributor-Andrew-orange)![Contributors](https://img.shields.io/badge/contributor-Luna-blue)![Contributors](https://img.shields.io/badge/contributor-Ayan-orchid)


## Getting Started

### Clone the Repository

```bash
git clone https://github.com/KUAS-ubicomp-lab/GlucoPatrol.git
cd GlucoPatrol
git checkout dev # Switch to branch 'dev'
```

### Setup Instructions


#### 1. Create the Virtual Environment and Install Dependancies:

#### Windows

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

If you get a script execution error,
run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
After that, restart the terminal and try activating the environment again.

</details>

##### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. Configure Environment Variables

After setting up the environment, copy the `.env.example` file located at the project root to a new `.env` and update the BASE_PATH variable to match your local project directory.

##### Windows

```bash
copy .env.example .env
```
##### macOS / Linux

```bash
cp .env.example .env
```
Open the newly created `.env` file and update the `BASE_PATH` variable to point to your local project directory.

```bash
BASE_PATH=C:/Users/HP/Downloads/GlucoPatrol  # Change this to your local path
DATA_SUBDIR=data/0_dataset
SEGDATA_SUBDIR=data/1_segmented_data
RAW_FEATURE_SUBDIR=features/0_raw
CLEAN_FEATURE_SUBDIR=features/1_clean
RESULTS_SUBDIR=results
```
#### 3. Set Up Model Training Files

Add your own `model_trainer_*.py` script in the `code/model_training/` folder to define custom training behavior.

#### 4. Configure Class Files

##### Windows
```bash
copy code/config/class_config.yaml.example code/config/class_config.yaml 
```
##### macOS / Linux
```bash
cp code/config/class_config.yaml.example code/config/class_config.yaml 
```
Open the newly created `class_config.yaml` file and update the `model_trainer` 's `module` to your custom  `model_trainer_*.py` script
```bash
# Default class configuration
data_loader:
  module: preprocessing.data_loader 
  class: DataLoader

data_segmenter:
  module: preprocessing.data_segmenter
  class: DataSegmenter

signal_processor:
  module: preprocessing.signal_processor
  class: SignalProcessor

feature_extractor:
  module: feature_engineering.feature_extractor
  class: FeatureExtractor

feature_cleaner:
  module: feature_engineering.feature_cleaner
  clasS: FeatureCleaner

model_trainer:
  module: model_training.model_trainer_FL # Change this to your custom model_trainer_*.py file
  class: ModelTrainer
```

#### 5. Run the FL Template Script
In `code/main_4_FL_Template.py`, customize the `folder_name` to specify your output directory and adjust `n_jobs` to match the number of processor cores you want to use for parallel processing.

```bash
model_trainer = ModelTrainer(
            results_dir=subject_results_dir, subject_id=subject_id, folder_name="XGBoost_Tuned", random_seed=42, n_jobs=18)
```

Finally, run:

##### Windows
```bash
python code/main_4_FL_Template.py
```
##### macOS / Linux
```bash
python3 code/main_4_FL_Template.py
```



