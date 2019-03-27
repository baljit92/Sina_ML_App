# Installtion

## Pre-Install
1. Clone the repo

2. Copy the weight file into data directory inside this repo, weights files will be mounted inside container at ```/data/<your_weights_name>.hdf5```

3. For annotation, copy the required folders

## Using Linux OS
```apt update 
apt install docker.io
sudo docker build -t app:1.0 . 
sudo docker run --name test1 -p 8080:8080 -d app:1.0
```
then access application on ```localhost:8080/ai/login```

## using Mac / windows

Install Docker as follows:

**MacOS:**  https://docs.docker.com/docker-for-mac/install/

**Microsoft Windows:** https://docs.docker.com/docker-for-windows/install/

On Windows, make sure the docker is running with ```Linux containers``` and ```Settings>Daemon>Experimental features is checked```. 
Finally restart the docker

Run the following commands from Powershell/Terminal 

```
docker build -t asksina_app:1.1 . 
docker run --name asksina -p 8080:8080 -d asksina_app:1.1
```
then access application on ```localhost:8080/ai/login```

# Mount volumes

if you want to mount another weights file then ( otherwise you don't need this step if you loaded weights earlier in docker build"

make sure weights are at /data on your machine so it can be mounted inside container

**on Linux/MacOS**
```sh sudo docker run --name test1 -v /data:/data -p 8080:8080 -d app:1.0```

**on Windows** 

if it is on ```"C:/data``` then ``` docker run --name test1 -v c:/data:/data -p 8080:8080 -d app:1.0```

otherwise replace ```"c:/data"``` with your other drive like ```"d:/data"```

# Using GPU on Linux
Setup CUDA and cudaNN

```apt-get install -y nvidia-container-runtime-hook``` <--- this is run time by nvidia

```sudo docker run --name test1 --runtime=nvidia -v /data:/data -p 8080:8080 -d app:1.0```
