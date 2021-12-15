FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7

COPY . /src/neuralpredictors
WORKDIR /src/neuralpredictors

RUN python3 -m pip install --upgrade pip &&\
    python3 -m pip install mypy==$(cat mypy_version.txt) &&\
    python3 -m pip install -e /src/neuralpredictors


ENTRYPOINT ["python3"]
