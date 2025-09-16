FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server python3.10-venv \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd


# Install dependencies

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt
