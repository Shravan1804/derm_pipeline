FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN cd /workspace && mkdir /workspace/code && mkdir /workspace/data && mkdir /workspace/logs

#RUN pip install icevision==0.5.2
#RUN pip install fastai==2.1.4

RUN pip install fastai

RUN pip install jupyter notebook jupyter_contrib_nbextensions scikit-learn scikit-image bidict efficientnet-pytorch dill pycocotools Shapely optuna p_tqdm


# FIX for opencv-python
# FIX for hanging tzdata
ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 screen -y
RUN pip install opencv-python
