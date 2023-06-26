FROM tensorflow/tensorflow:latest-gpu

COPY ./API/main.py .
COPY ./API/inference.py .
COPY ./Models/modelGPT ./Models/modelGPT
COPY ./Models/modelDiffusion ./Models/modelDiffusion
COPY ./API/index.html .
COPY ./API/lyrics.html .
COPY requirements.txt .

RUN pip install -r requirements.txt


CMD ["python", "./main.py"]