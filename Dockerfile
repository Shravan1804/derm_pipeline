FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# FIX for opencv-python
# FIX for hanging tzdata
ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install git ffmpeg libsm6 libxext6 screen -y

RUN pip install --upgrade pip

WORKDIR /workspace

COPY . /workspace/
RUN git config --global --add safe.directory '*'

RUN pip install -r /workspace/requirements.txt --upgrade --no-cache-dir
RUN pip install -r /workspace/requirements_dev.txt --upgrade --no-cache-dir

ENV PYTHONPATH "$PYTHONPATH:./"
ENV NUMBA_CACHE_DIR=/numba_cache_dir
ENV MPLCONFIGDIR=/mpl_cache_dir
