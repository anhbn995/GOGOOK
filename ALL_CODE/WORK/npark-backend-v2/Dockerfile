FROM ubuntu:20.04

ENV CONDA_DIR /opt/conda

RUN apt-get update && apt-get -y upgrade && apt-get install -y gcc g++ curl libgl1

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > ~/miniconda.sh  && \
  /bin/bash ~/miniconda.sh -b -p $CONDA_DIR

ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app

COPY environment.yml environment.yml

RUN conda env create -f environment.yml

ENV PATH $CONDA_DIR/envs/npark/bin:$PATH

COPY skyamqp-0.0.1-py3-none-any.whl .

RUN /bin/bash -c "source activate npark && pip install skyamqp-0.0.1-py3-none-any.whl"

COPY . .

CMD python scheduler.py

