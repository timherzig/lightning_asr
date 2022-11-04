FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc python3-pip ffmpeg && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir --user -r requirements.txt

