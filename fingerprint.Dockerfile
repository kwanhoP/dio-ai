FROM python:3.7
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install \
    gcc nano \
    ffmpeg libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 -y

WORKDIR /code
RUN pip install --upgrade pip
RUN pip install git+https://github.com/worldveil/dejavu.git

COPY fingerprint.py /code

ENTRYPOINT ["python3", "fingerprint.py"]