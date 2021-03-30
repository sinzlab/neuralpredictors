FROM eywalker/pytorch-jupyter:v0.4.1-cuda9.2


RUN apt-get -y update && apt-get  -y install ffmpeg libhdf5-10 git python3-pip
RUN pip3 install imageio ffmpy h5py opencv-python statsmodels datajoint

RUN pip3 install torch torchvision datajoint --upgrade

RUN pip install jupyterlab && \
    jupyter serverextension enable --py jupyterlab --sys-prefix

ADD . /src/ml-utils
RUN pip install -e /src/ml-utils

WORKDIR /src

WORKDIR /notebooks

RUN mkdir -p /scripts
ADD jupyter/run_jupyter.sh /scripts/
ADD jupyter/jupyter_notebook_config.py /root/.jupyter/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter.sh"]
