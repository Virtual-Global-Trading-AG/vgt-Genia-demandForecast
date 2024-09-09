FROM python:3.11

WORKDIR /app

COPY demandForecastDataManager.py /app/demandForecastDataManager.py
COPY main.py /app/main.py
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]