FROM pytorch/pytorch

# Install dependencies

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Set default command
CMD ["bash"]
