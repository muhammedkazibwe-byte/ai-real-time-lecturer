import streamlit as st

st.set_page_config(page_title="Test App", layout="wide")

st.title("🎓 Quick Test – AI Lecturer MVP")

st.write("If you see this → Streamlit is working!")

name = st.text_input("Enter your name", key="name_input")

if st.button("Say hello", key="hello_btn"):
    st.success(f"Hello {name or 'friend'}! App is alive 🚀")

