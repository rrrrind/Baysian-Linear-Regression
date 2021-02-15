FROM ubuntu:18.04

RUN apt update && apt -y upgrade
RUN apt install -y python3 python3-pip

WORKDIR /home
COPY requirements.txt ${PWD}
RUN pip3 install -r requirements.txt

# 作業ディレクトリ
WORKDIR /home/workspace/