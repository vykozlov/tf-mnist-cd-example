FROM tensorflow/tensorflow:1.5.0-gpu

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
        && \
    python -m ipykernel.kernelspec && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

# Set the working directory to /home/apps
WORKDIR /home/apps

# Copy the current directory contents into the container at /home
ADD . /home

# Run mnist_deep.py when the container launches. BETTER TO AVOID THIS!
#CMD ["python", "mnist_deep.py"]
