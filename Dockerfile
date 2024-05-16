FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

# FIX for opencv-python
# FIX for hanging tzdata
ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install git ffmpeg libsm6 libxext6 screen -y

RUN pip install --upgrade pip

WORKDIR /workspace

RUN mkdir /assets

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt --upgrade --no-cache-dir

COPY requirements_dev.txt /assets/requirements_dev.txt
RUN pip install -r /assets/requirements_dev.txt --upgrade --no-cache-dir

COPY . /workspace/
RUN git config --global --add safe.directory '*'

ENV PYTHONPATH "$PYTHONPATH:./"
ENV NUMBA_CACHE_DIR=/numba_cache_dir
ENV MPLCONFIGDIR=/mpl_cache_dir
