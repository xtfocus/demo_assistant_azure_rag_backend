FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 3100

ENTRYPOINT ["gunicorn", "-c", "src/gunicorn.conf.py", "src:create_app()"]
