# Heart Stroke Prediction Project

This project is a web application for predicting the risk of heart disease using a logistic regression model. The application is built using Streamlit and provides an interactive interface for users to input their health data and receive predictions.

## Project Structure

```
heartprediction
├── app.py                     # Main application file for heart stroke prediction
├── requirements.txt           # Python dependencies required for the project
├── .gitignore                 # Files and directories to be ignored by Git
├── models                     # Directory containing model files
│   ├── Logistic_Regression.pkl # Serialized logistic regression model
│   ├── scaler.pkl             # Serialized scaler for input feature standardization
│   └── columns.pkl            # Expected feature names for the model
├── data                       # Directory for storing raw data files
│   └── raw                    # Raw data files used for training/testing
├── notebooks                  # Directory for Jupyter notebooks
│   └── exploratory.ipynb      # Notebook for exploratory data analysis
├── src                        # Source code directory
│   ├── utils.py               # Utility functions for various tasks
│   └── preprocessing.py       # Functions for preprocessing input data
├── tests                      # Directory for unit tests
│   └── test_app.py            # Unit tests for app.py functionality
└── README.md                  # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd heartprediction
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

- Open the web application in your browser.
- Provide the required health details using the input fields.
- Click on the "Predict" button to receive the prediction result indicating the risk of heart disease.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.