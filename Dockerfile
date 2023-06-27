FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y git-lfs
RUN git clone https://github.com/RecklessFox/exerciseAI.git
RUN mv exerciseAI/Models .
RUN mv exerciseAI/* .


RUN pip install -r requirements.txt
