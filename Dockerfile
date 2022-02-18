FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7

COPY . /src/neuralpredictors
WORKDIR /src/neuralpredictors

RUN python3 -m pip install --upgrade pip &&\
    python3 -m pip install --requirement /src/neuralpredictors/requirements-dev.txt  &&\
    python3 -m pip install --no-use-pep517 -e /src/neuralpredictors


ENTRYPOINT ["python3"]
