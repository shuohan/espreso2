FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
RUN apt-get update && apt-get install -y git
RUN pip install scipy==1.6.3 \
        nibabel==3.2.1 \
        matplotlib==3.4.2 \
        tqdm==4.51.0 \
        improc3d==0.5.2 \
        scikit-image==0.18.1 \
        git+https://github.com/shuohan/espreso2@0.2.2
ENV MPLCONFIGDIR=/tmp/matplotlib
CMD ["bash"]
