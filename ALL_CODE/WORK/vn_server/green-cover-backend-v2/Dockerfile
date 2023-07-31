FROM continuumio/miniconda3
RUN echo "source activate base" > ~/.bashrc
ENV PATH /opt/conda/bin:$PATH
WORKDIR /code
COPY . /code
RUN apt-get -y update
RUN apt-get -y install libpq-dev
RUN apt-get -y install build-essential
RUN apt-get -y install clang cmake
RUN apt-get install libmagickwand-dev
RUN conda install gdal opencv tensorflow-gpu=1.13.1 keras ipywidgets numba
RUN jupyter nbextension enable --py widgetsnbextension
RUN pip install tensorrtserver-1.6.0-py2.py3-none-linux_x86_64.whl
RUN pip install -r requirements.txt

EXPOSE 5002"./app/controllers/imagery_controller.py"
CMD ["python", "manage.py runserver"]
RUN CC=`which clang` pip install uwsgi
RUN sudo docker run -d --rm --gpus 2 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p7000:8000 -p7001:8001 -p7002:8002 -v/home/halh/tensorrt-server/model_repository:/models nvcr.io/nvidia/tensorrtserver:19.08-py3 trtserver --model-store=/models
RUN //192.168.1.120/geoaistorage /home/geoai/geoai_data cifs credentials=/root/.smbcredentials,vers=1.0,iocharset=utf8,gid=1000,uid=1000,file_mode=0777,dir_mode=0777 0 0
RUN //192.168.1.120/datacubestorage /home/geoai/datacube_data cifs credentials=/root/.smbcredentials,vers=1.0,iocharset=utf8,gid=1000,uid=1000,file_mode=0777,dir_mode=0777 0 0
RUN sudo nvidia-smi -pm 1
RUN sudo nvidia-smi -pl 240