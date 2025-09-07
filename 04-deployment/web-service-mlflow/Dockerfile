FROM python:3.13-slim

RUN pip install -U pip

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ["predict.py", "ridge_reg.bin", "./"]

EXPOSE 9696

CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]