ARG NGC_VERSION="25.05-py3"

FROM nvcr.io/nvidia/pytorch:$NGC_VERSION 
COPY . /workspace/MultiMeditron

WORKDIR /workspace/MultiMeditron
RUN pip install -e .

