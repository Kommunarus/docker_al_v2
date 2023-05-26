FROM nvidia/cuda:11.6.2-base-ubuntu20.04
CMD nvidia-smi
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y && apt-get install -y python3-pip python3.10 build-essential libgl1-mesa-glx ffmpeg libsm6  \
    libxext6 git

COPY . .

EXPOSE 5000
ENV FLASK_APP 'app.py'
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV GIT_PYTHON_REFRESH 'quiet'

RUN pip3 install --upgrade pip &&  \
    pip3 install -r requirements.txt &&  \
    pip3 install -r /custom_model/requirements.txt

ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0"]
