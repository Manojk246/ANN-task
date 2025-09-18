import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

# ---------------------------
# Load and preprocess dataset
# ---------------------------
# Adjust the file path if needed
try:
    df = pd.read_csv("modified.csv")
except FileNotFoundError:
    print("The file 'modified.csv' was not found.")
    exit()

# Separate features (X) and target (y)
X = df.drop(columns=['quality'])
y = df['quality']
feature_names = X.columns.tolist()

# Train-test split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# ANN Model Definition
# ---------------------------
def build_ann(hidden_units=64, learning_rate=0.001, dropout_rate=0.2, n_features=None):
    model = Sequential()
    model.add(Dense(hidden_units, activation="relu", input_shape=(n_features,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units // 2, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model

# ---------------------------
# Pipeline and GridSearchCV
# ---------------------------
select_k = 10 
pipe = Pipeline([
    ("power", PowerTransformer(method="yeo-johnson")),
    ("scale", RobustScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("select", SelectKBest(score_func=f_regression, k=select_k)),
    ("ann", KerasRegressor(
        model=build_ann,
        model__n_features=select_k,
        verbose=0,
        batch_size=32,
        epochs=50
    ))
])

param_grid = {
    "ann__model__hidden_units": [64],
    "ann__model__learning_rate": [0.001],
    "ann__model__dropout_rate": [0.1],
    "ann__batch_size": [32],
    "ann__epochs": [50],
}

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="r2",
    cv=3,
    n_jobs=-1
)

print("Starting model training...")
search.fit(X_tr, y_tr)
print("Model training complete.")

# ---------------------------
# Prediction Function for Gradio
# ---------------------------
def predict_quality(*features):
    input_df = pd.DataFrame([features], columns=feature_names)
    quality_pred = search.predict(input_df)[0]
    return f"The predicted wine quality is: {quality_pred:.4f}"

# ---------------------------
# Login Validation
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
            gr.Markdown("## 🔑 Login to Access Wine Quality Prediction App")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Textbox(label="Login Status")

    # Page 2: Prediction App
    with gr.Row(visible=False) as app_page:
        with gr.Column():
            gr.Markdown("## 🍷 Wine Quality Prediction App")
            
            inputs = []
            with gr.Accordion("Enter Feature Values", open=True):
                for col in feature_names:
                    inputs.append(gr.Number(label=col, value=float(df[col].median())))

            btn = gr.Button("Predict Quality")
            output_text = gr.Textbox(label="Predicted Quality")

    # Button Actions
    login_btn.click(fn=login, inputs=[username, password], outputs=[login_page, app_page])
    btn.click(fn=predict_quality, inputs=inputs, outputs=output_text)

# ---------------------------
# Launch
# ---------------------------
demo.launch(share=False)