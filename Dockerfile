FROM ubuntu
RUN apt-get update
RUN apt-get install python3-pip -y
COPY . /app
COPY data /data
WORKDIR /app
RUN pip3 install  --no-cache-dir  -r requirements.txt
RUN apt-get install libsm6 libxrender1 libfontconfig1 libxext6 -y
WORKDIR src/webapp/TrainImage_Annotate/
RUN python3 manage.py makemigrations
RUN python3 manage.py migrate
CMD python3 manage.py runserver 0.0.0.0:8080
