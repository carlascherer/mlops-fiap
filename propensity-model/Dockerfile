FROM python:3.12
LABEL maintainer="Carla Scherer"
LABEL version="1.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app
RUN mkdir -p /log

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip

RUN ls -la /app

RUN cat /app/requirements.txt

RUN sed -i '/pywin32/d' /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

VOLUME /app/config
VOLUME /log

EXPOSE 5000

CMD ["python", "inferencia.py"]
