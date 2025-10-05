FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH="/app:/app/src"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY ./src ./src
COPY ./data ./data
COPY ./indexes ./indexes

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
