
FROM python:3.7

RUN pip install --upgrade pip

WORKDIR /home/app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y 
RUN apt install tesseract-ocr
RUN apt install libtesseract-dev

COPY . /home/app

RUN pip install -r requirements.txt
RUN python -m spacy download en

EXPOSE 4000

#CMD ["python","app.py"]
#CMD ["python","app_api.py"]
CMD ["python","app_webapi.py"]

