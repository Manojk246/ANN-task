🍷 ANN Regression App — Wine Quality Prediction

This is a Gradio-based web application that predicts Wine Quality using an Artificial Neural Network (ANN).
It provides an interactive UI where you can input feature values, choose training options, and visualize results.

🚀 Features

🔑 Login Authentication (admin: admin, password: 1234)

🤖 ANN Regression Model built using TensorFlow/Keras

📊 Interactive Gradio UI for input & prediction

🔄 Training & Evaluation with live feedback

🎨 Data Visualization (Loss curves, Predictions vs. True values)

📁 Modular Code (app.py) — ready to deploy


📦 Requirements

The main libraries used:

gradio
pandas
numpy
matplotlib
scikit-learn
tensorflow

▶️ Usage

Run the app:

python app.py


Gradio will show a link in the terminal, for example:

Running on local URL:  http://127.0.0.1:7860


Open the link in your browser to access the app.

🔑 Login Credentials

Username: admin

Password: 1234

📊 Example Workflow

Login using the provided credentials

Input Wine chemical properties (like alcohol, sulphates, pH, etc.)

Select Train & Predict

View:

✅ Predicted Wine Quality

📉 Training Loss Curve

📈 Predictions vs True Quality

🚀 Deployment

You can deploy the app on:

Hugging Face Spaces

Streamlit Cloud

Google Colab (with ngrok)

Example for Hugging Face:

pip install huggingface_hub


Push your repo and define app.py as the entry file.

✨ Future Improvements

Add Dropout & BatchNorm for better generalization

Allow custom dataset upload

Add more ML models (Random Forest, XGBoost) for comparison