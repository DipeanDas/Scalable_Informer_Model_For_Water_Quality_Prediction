<h1>Scalable Informer Model For Water Quality Prediction</h1>

This repository contains the implementation of a **Scalable Informer-based deep learning architecture** for **water quality forecasting**. The model was primarily trained and tested on **multivariate water quality data** collected from various locations of the **Ganges River**. Although demonstrated on the Ganges dataset, the architecture is **generalizable**, **customizable** and can be applied to **other river systems or water quality monitoring datasets**. 
An interactive **Flask-based web interface** is also developed to deploy the trained model for real-time prediction and visualization.

##📊Training Dataset Overview:
 - Source: Ganges River water quality dataset.
 - Time Period: 2022-2024
 - Locations Count: 15
 - Water Quality parameters count: 23[BOD, COD, DO, pH, TDS,TC,FC,Nitrate and others]
 - Contextual parameters count: 3[Location. Year, Month]
 - Target Parameter: Biochemical Oxygen Demand (BOD). 
 - The last column is the prediction target (BOD), while all other columns are input features.
   
**NOTE:** The dataset is formatted in a generic manner so that users can replace it with their own dataset.Guidance for customizing data handling is provided via commented instructions inside **custom_data_process.py** and **data_loader.py** file.

## 📂 Project Structure
The model codebase is structured in the following way:
```
WQA_Informer_Model/
│
├── data/
│ ├── Water_quality.csv # Dataset (Locationwise monthly data of water quality parameters)
│ ├── scaler_p.py # codefile to scale data for training
│
├── exp/
│ └── exp_informer.py # Experiment pipeline: training, validation, testing
│
├── models/
│ ├── informer.py # Core Informer model integration
│ ├── encoder.py # Encoder layers (self-attention, conv, feedforward)
│ ├── decoder.py # Decoder layers (masked + cross attention)
│ ├── embed.py # Embedding layers (temporal, positional, data embedding)
│ └── attn.py # ProbSparse & Full Attention mechanisms
│
├── utils/
│ ├── tools.py # Metrics (MSE, MAE, R², PLCC, SRCC and KRCC)
│ ├── data_loader.py # Data preprocessing + DataLoader for train/val/test
│ └── custom_data_process.py # CSV preprocessing (scaling, feature extraction)
│
├── score_metrics/
│ └── metrics.txt # Saved test scores & evaluation metrics
│
├── static/
│ └── styles.css # CSS styling for Flask web app
│
├── templates/
│ ├── index.html # Home page template
│ └── predict.html # Prediction page template
│
├── config.py # Configurations (Model  training and testing parameter value setup)
├── main.py # Entry point: Initiates training process
├── pred_app.py # Prediction pipeline/system logic
├── app.py # Flask application entry point (integrates pred_app with UI)
├── requirements.txt # Python libraries and dependencies
└── README.md # Project documentation (this file)
```
## ⚙️ Dependencies

This project was developed and tested with the following dependencies:
```
- Python 3.10  
- torch >= 1.10.0  
- numpy >= 1.21.0  
- pandas >= 1.3.0  
- scikit-learn >= 1.0.0  
- matplotlib >= 3.4.0  
- joblib >= 1.0.0  
- openpyxl >= 3.0.0  
- python-dateutil >= 2.8.0  
- scipy >= 1.7.0  
- flask >= 2.0.0  
```
## ⚙️ Installation
Clone the repository and install the required dependencies:
```
git clone https://github.com/DipeanDas/Scalable_Informer_Model_For_Water_Quality_Prediction.git
cd Scalable_Informer_Model_For_Water_Quality_Prediction
pip install -r requirements.txt 
```
## 🚀 Usage
###1️⃣ Training the Informer Model

After customizing and setting up configuration parameter values, run the training pipeline using:
```
python main.py
```
This will:
- Load the dataset
- Train the Informer model
- Validate on held-out data
- Save the best model checkpoint under ./checkpoints/
- Metrics such as MSE, MAE, RMSE, R², PLCC, SRCC, KRCC will be reported in txt file.

###2️⃣ Running the Web Application

Once the training is complete and trained model is saved, launch the Flask app:
```
python app.py
```
Here:
- The app connects to the trained Informer model.
- Provides an interactive interface for prediction.

**NOTE:** UI elements and features can be customized easily within app.py and templates files as per requirement.

## 🌍 Applications

- Scalable to other rivers or aquatic systems.
- Predict different water quality parameters (by adjusting target column).
- Useful for environmental monitoring, policy-making, and decision support systems.

##👨‍💻 Contributors

**Dipean Dasgupta** (Department of CSE, IIIT Vadodara, India)
**Bishnu Prasad Sahoo** (Forest Ecology and Climate Change Division, Forest Research Institute, Dehradun, India)
**Pramit Mazumdar** (Department of CSE, IIIT Vadodara, India)
