# Disease Prediction App

This repository contains the code and resources for a **Disease Prediction Web App** built using **Streamlit** and a **Logistic Regression** model. The app predicts potential diseases based on a user-selected set of symptoms.

## Features

- **User-Friendly Interface**: Built with Streamlit for a smooth and responsive experience.
- **Disease Prediction**: Using a trained Logistic Regression model to predict diseases based on selected symptoms.
- **Metrics Display**: Displays accuracy, precision, recall, and other relevant metrics of the model during predictions.
- **Model Description**: Includes detailed information on the trained model and its accuracy metrics.

## Technologies Used

- **Streamlit**: A framework to build interactive web applications for machine learning projects.
- **scikit-learn**: For machine learning model training and prediction.
- **numpy**: For numerical computation.
- **joblib**: To load the pre-trained Logistic Regression model.
- **matplotlib**: Optional, used for data visualization (e.g., to plot model metrics).

## How to Run the App Locally

1. **Clone the repository**:
    ```bash
    git clone https://github.com/azeemanwar789/DIsease-Prediction-Model.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd DIsease-Prediction-Model
    ```

3. **Install dependencies**:
    Create a virtual environment and install required libraries using:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**:
    ```bash
    streamlit run Model2/UI_Week_7.py
    ```

    This will start the Streamlit app, and you can access it in your browser at `http://localhost:8501`.

## Model Details

The app uses a pre-trained **Logistic Regression** model (`lr_disease_prediction_model.joblib`). This model was trained on various health symptoms and their respective diseases. The model predicts the most likely disease based on the symptoms the user selects.

## Metrics

The app shows key performance metrics of the model:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

These metrics help in understanding how well the model performs during disease predictions.

## Disease Categories

The model can predict the following diseases:
- (vertigo) Paroxysmal Positional Vertigo
- AIDS
- Acne
- Alcoholic Hepatitis
- Allergy
- Arthritis
- Bronchial Asthma
- Cervical Spondylosis
- Chicken Pox
- Common Cold
- ... *(and many more!)*

See the full list in the `disease_names` dictionary within the code.

## App Path

The application can be found at the following path in the repository:  
[Model2/UI_Week_7.py](https://github.com/azeemanwar789/DIsease-Prediction-Model/blob/8b8c709fa6faafff64b7caf37ae98aa6f033d678/Model2/UI_Week_7.py)

## Contributing

If you'd like to contribute to this project, feel free to fork this repository and submit a pull request with your changes. All contributions are welcome!

## License

This project is open source and available under the MIT License.
