import os
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")


def render_form():
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Ticket Class", [1, 2, 3], index=2)
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"], index=0)
        fare = st.number_input("Fare", min_value=0.0, value=8.05, step=0.5)

    return {"Pclass": pclass, "Sex": sex, "Age": age, "Fare": fare}


def call_predict(payload):
    r = requests.post(f"{API_URL}/predict", json={"features": payload}, timeout=15)
    r.raise_for_status()
    return r.json()


def main():
    st.set_page_config(page_title="Titanic Survival", page_icon="🚢")
    st.title("🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢 Will You Survive The Titanic? 🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢🚢")

    features = render_form()

    if st.button("Predict Survival", type="primary"):
        data = call_predict(features)
        pred = int(data.get("prediction", 0))
        prob = float(data.get("probability", 0.0))

        if pred == 1:
            st.success(f"🎉 Survived! Probability {prob:.2%}")
        else:
            st.error(f"💀 Did not survive. Probability {(1 - prob):.2%}")


if __name__ == "__main__":
    main()
