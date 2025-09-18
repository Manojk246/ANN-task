import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("modified.csv")

X = df.drop("quality", axis=1).values
y = df["quality"].values

# Encode labels (wine qualities: 3–8)
classes = sorted(df["quality"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_idx[val] for val in y])

num_classes = len(classes)
y_cat = to_categorical(y, num_classes=num_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Build ANN model
# ---------------------------
def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    return model

model = build_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# ---------------------------
# Prediction function
# ---------------------------
def predict_wine_quality(*features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    probs = model.predict(features_scaled, verbose=0)[0]
    pred_class = classes[np.argmax(probs)]
    confidence = np.max(probs)
    return f"✅ Predicted Wine Quality: {pred_class} (Confidence: {confidence:.2f})"

# ---------------------------
# Login validation
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
            with gr.Accordion("Enter Feature Values", open=False):
                for col in df.drop("quality", axis=1).columns:
                    inputs.append(gr.Number(label=col, value=float(df[col].median())))

            btn = gr.Button("Predict Quality")
            output_text = gr.Textbox(label="Result")

    # Button actions
    login_btn.click(fn=login, inputs=[username, password], outputs=[login_msg, app_page])
    btn.click(fn=predict_wine_quality, inputs=inputs, outputs=output_text)

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch()
