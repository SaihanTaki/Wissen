FROM python:3.9-slim

WORKDIR /app

COPY ./requirements.txt .

RUN pip install -r requirements.txt

RUN mkdir -p /app/src

COPY ./pdfai/ ./src/

# ENV $(cat .env)

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/app.py"]