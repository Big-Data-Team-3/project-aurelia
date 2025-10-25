import streamlit as st

def message_bubble(role, message):
    if role == "user":
        st.markdown(f"<div style='background:#2e86de;color:white;padding:10px;border-radius:10px;margin:5px 0;text-align:right'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#f1f2f6;padding:10px;border-radius:10px;margin:5px 0;text-align:left'>{message}</div>", unsafe_allow_html=True)
