FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update && apt-get install -y git graphviz 
RUN pip install scipy==1.5.4 \
        nibabel==3.2.0 \
        matplotlib==3.3.3 \
        graphviz==0.14.2 \
        git+https://github.com/shuohan/sssrlib@0.2.0 \
        git+https://github.com/shuohan/singleton-config@0.2.0 \
        git+https://github.com/shuohan/pytorchviz@0.0.2 \
        git+https://github.com/shuohan/ptxl@0.2.0 \
        git+https://github.com/shuohan/espreso2@0.1.1
ENV MPLCONFIGDIR=/tmp/matplotlib
CMD ["bash"]
