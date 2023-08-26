FROM python:3.10.13
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
#EXPOSE $PORT
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]