FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
RUN apt-get update && apt-get install -y git graphviz 
# RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
RUN pip install scipy==1.5.4 \
        nibabel==3.2.0 \
        matplotlib==3.3.3 \
        graphviz==0.14.2 \
        git+https://github.com/shuohan/sssrlib@0.2.0 \
        git+https://github.com/shuohan/singleton-config@0.2.0 \
        git+https://github.com/shuohan/pytorchviz@0.0.2 \
        git+https://github.com/shuohan/ptxl@0.2.0
COPY . /tmp/
RUN pip install /tmp && rm -rf /tmp/*
ENV MPLCONFIGDIR=/tmp/matplotlib
CMD ["bash"]
