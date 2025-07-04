import streamlit as st
import requests

st.set_page_config(page_title="Fitness & Diet Assistant", layout="centered")

st.title("🧘‍♂️ Diet & Workout Assistant (India-focused)")

with st.form("query_form"):
    question = st.text_area("Ask a question (diet, fitness, recipes...)", height=120)
    goal = st.selectbox("Your health goal", ["General health", "Lose weight", "Gain weight", "Build muscle"])
    ingredients_input = st.text_input("Available ingredients (comma-separated)")
    submitted = st.form_submit_button("Ask Assistant")

if submitted:
    with st.spinner("Thinking..."):
        ingredients = [i.strip() for i in ingredients_input.split(",") if i.strip()]
        payload = {
            "question": question,
            "goal": goal,
            "ingredients": ingredients
        }
        try:
            res = requests.post("http://localhost:8000/ask-assistant", json=payload)
            response = res.json().get("response", "Something went wrong.")
        except Exception as e:
            response = f"⚠️ API Error: {str(e)}"

    st.markdown("### 🧠 Assistant says:")
    st.write(response)
