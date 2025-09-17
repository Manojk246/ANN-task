# app.py
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv("winequality-red.csv")  # <-- replace with your file name
X = data.drop(columns=["quality"])
y = data["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Build ANN model
# ---------------------------
model = Sequential([
    Input(shape=(X_train.shape[1],)),   # Input layer
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")       # Regression output
])

model.compile(optimizer="adam",
              loss="mse",
              metrics=["mae"])

# Train model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

# Evaluate on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

eval_text = f"📊 Model Performance:\nMSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}"

# ---------------------------
# Prediction function
# ---------------------------
def predict_quality(*features):
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0][0]
    return f"🍷 Predicted Wine Quality: {pred:.2f}\n\n{eval_text}"

# ---------------------------
# Login function
# ---------------------------
def login(username, password):
    if username == "admin" and password == "1234":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(value="❌ Invalid login. Try again."), gr.update(visible=False)

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as demo:
    # Page 1: Login
    with gr.Row(visible=True) as login_page:
        with gr.Column():
            gr.Markdown("## 🔑 Login to Access Wine Quality Predictor")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Textbox(label="Login Status")

    # Page 2: ANN App
    with gr.Row(visible=False) as app_page:
        with gr.Column():
            gr.Markdown("## 🍷 Wine Quality Prediction using ANN")

            inputs = []
            with gr.Accordion("Enter Wine Features", open=True):
                for col in X.columns:
                    inputs.append(gr.Number(label=col, value=float(data[col].median())))

            btn = gr.Button("Predict Quality")
            output_text = gr.Textbox(label="Result")

    # Button Actions
    login_btn.click(fn=login, inputs=[username, password], outputs=[login_msg, app_page])
    btn.click(fn=predict_quality, inputs=inputs, outputs=[output_text])

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch()
