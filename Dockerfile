FROM nvcr.io/nvidia/pytorch:23.05-py3
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 xvfb 
WORKDIR /workspace
COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN python totalsegmentator/download_pretrained_weights.py
