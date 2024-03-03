FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

RUN chmod 1777 /tmp && chmod 1777 /var/tmp


RUN apt-get update
RUN apt-get upgrade -y 
RUN apt-get install -y vim curl ca-certificates amqp-tools wget graphviz

RUN apt-get install -y python3
RUN apt-get install -y python3-pip

RUN pip install --upgrade protobuf==3.20.1
RUN pip install termcolor
RUN pip install pandas
RUN pip install numpy
RUN pip install torch==2.0.1
RUN pip install transformers
RUN pip install scikit-learn
RUN pip install matplotlib
RUN pip install lightning
RUN pip install wandb
RUN pip install torchvision==0.15.2
