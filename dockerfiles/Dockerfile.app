FROM python:3.9-slim-buster

RUN apt-get update -y && apt-get install -y python3-pip python3-dev python3

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 5000

ENTRYPOINT ["python3", "app.py"]