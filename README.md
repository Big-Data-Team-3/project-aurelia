How to run:

cd app/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

cd app/streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0