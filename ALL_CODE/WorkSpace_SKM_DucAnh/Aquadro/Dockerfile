FROM registry.eofactory.ai:5000/ubuntu2204-conda

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

COPY . .

SHELL ["/bin/bash", "-c"]

CMD source /opt/conda/etc/profile.d/conda.sh && conda activate monitor && uwsgi --ini uwsgi/docker-uwsgi.ini


