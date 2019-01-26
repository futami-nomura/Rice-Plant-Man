FROM python:3.6

ARG project_dir=/app/
COPY . $project_dir

RUN apt-get update
RUN apt-get install -y vim

RUN python -m pip install --upgrade pip
RUN python -m pip install image
RUN python -m pip install flask
RUN python -m pip install keras
RUN python -m pip install tensorflow
RUN python -m pip install numpy
