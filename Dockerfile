FROM tensorflow/tensorflow:1.7.0-gpu

LABEL maintainer="Valentin Kozlov <valentin.kozlov@kit.edu>"
# Dockerfile based on the one for Tensorflow from Tensorflow:
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker
# modified by Valentin Kozlov on 21-Mar-2018

RUN apt-get update && apt-get install -y --no-install-recommends git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
# python
    pip --no-cache-dir install \
        keras \
        jupyterlab \
        pylint \
        && \
    python -m ipykernel.kernelspec && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

# For Jupyter terminal
ENV SHELL /bin/bash

# Set the working directory to /home/apps
WORKDIR /home/apps

# Set up our notebook config.
COPY jupyter/jupyter_notebook_config.py /root/.jupyter/
COPY jupyter/run_jupyter.sh /

# REMINDER: Tensorflow Docker Images EXPOSEs 6006 and 8888 ports

# Copy the current directory contents into the container at /home
ADD . /home
