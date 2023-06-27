FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
ENV PYTHONPATH /workspace

RUN apt-get update && apt-get install --no-install-recommends -y ffmpeg libsm6 libxext6 xvfb git wget build-essential

# NOTE: install dependencies through conda to keep docker setup as close to local setup as possible
SHELL ["/bin/bash", "--login", "-c"]
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -u -p /miniconda && \
    /miniconda/bin/conda init bash
ENV PATH="/miniconda/bin:${PATH}"

COPY env.yaml .
RUN conda install -y -c conda-forge mamba; mamba env update -n base -f env.yaml

COPY requirements.txt  .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .
RUN totalsegmenter download-weights
