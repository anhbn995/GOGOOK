FROM registry.eofactory.ai:5000/eof/base-images/tf2.7

WORKDIR /app

RUN pip install geopandas scikit-image scikit-learn 



COPY ./model ./model
COPY ./Vectorization.cpython-38-x86_64-linux-gnu.so ./Vectorization.cpython-38-x86_64-linux-gnu.so
COPY ./*.py ./

CMD ["python", "predict_farm.py"]