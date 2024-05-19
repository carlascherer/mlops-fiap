FROM python:3.12
LABEL maintainer="Carla Scherer"
LABEL version="1.0"

RUN apt-get update && apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /myServer
RUN mkdir -p /log

COPY . /myServer

WORKDIR /myServer

RUN pip install --upgrade pip

RUN ls -la /myServer

RUN [ -f requirements.txt ] && echo "requirements.txt found" || echo "requirements.txt not found"

RUN sed -i '/pywin32/d' requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

VOLUME /myServer/config
VOLUME /log

EXPOSE 8080

CMD ["python", "modelmanager.py"]
