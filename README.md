# System Architecture 

The proposed system is structured into three layers: Data Collection Layer, Processing & Modelling Layer, and User Interface Layer.

### Data Collection Layer
The Empatica EmbracePlus smartwatch will be used to collect high-resolution, multimodal wearable data. This device is equipped with sensors for continuous physiological monitoring, including heart rate, SpO₂, electrodermal activity, and movement. Lightweight and widely used in clinical research and remote patient monitoring, the EmbracePlus allows users to access session data via Empatica’s Care Portal. Users can export the required session data and upload it through the User Interface Layer.

### Processing & Modelling Layer
This layer consists of the backend of the web application, particularly a data preparation pipeline and the non-invasive glucose prediction models. Whenever user-uploaded data is sent to the backend, the data will be cleaned using a preparation pipeline and features will be extracted.  These features will be used as inputs to the glucose prediction models to generate a predicted glucose value. 

The models are trained and validated offline using a public dataset containing simultaneously recorded CGM and wearable data from the EmbracePlus predecessor, the Empatica E4. The modeling process follows a structured pipeline comprising data preprocessing, feature engineering, model training/tuning, validation and testing.

### User Interface Layer
This layer comprises the frontend of the web application, allowing users to upload data exported from the Care Portal. Once uploaded, the data is processed by the backend, and the resulting glucose profile is displayed along with additional metrics such as Time in Range (TIR) and the Glucose Management Indicator (GMI).

# Predictive Modelling using CI/AI Techniques

We adopt a bottom-up modeling approach to account for the nested structure of the dataset. Individual models are first trained using data from each subject, followed by the development of a meta-model that ensembles these individual models.

Specifically, the dataset is divided into 10 subjects for training and 6 subjects for testing. Ten individual models are trained using the data from each subject in the training set. These models are then aggregated into a meta-model, which is evaluated on the 6 test subjects. The final model performance is calculated as the average across these 6 subjects.

Model performance will be evaluated comprehensively using the following metrics:
* R-squared (R²)
* Root Mean Squared Error (RMSE)
* Normalized RMSE (NRMSE)
* Bland-Altman (BA) Plots
* Clarke Error Grid (CEG)

In addition, SHAP will be used to interpret the contribution of each feature.

# Target Users and Usage Scenarios
* Diabetes Patients: For self-management and reducing the risk of complications.
* Healthcare Professionals and Care Providers: To assess long-term glucose fluctuations and recommend lifestyle interventions based on glucose trends.
* Pre-diabetic and Normoglycemic Individuals: To support lifestyle optimization for diabetes prevention. The system enables regular, non-invasive, and cost-effective glucose monitoring, lowering the barriers to managing glucose levels and overall metabolic health.

### Usage Scenarios
The GlucoPatrol system is designed for daily use at home, in the workplace, or in other routine environments as a non-invasive alternative to traditional glucose monitoring methods. It operates under free-living conditions without requiring manual data logging. Users simply upload their wearable data via an internet connection to access their glucose profiles.

