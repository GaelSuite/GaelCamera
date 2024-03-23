FROM python:3.8-slim

WORKDIR /app

RUN pip install --no-cache-dir roboflow ultralytics numpy matplotlib

COPY train.py /app/

CMD ["python", "train1.py"]
