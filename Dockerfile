FROM ubuntu:latest
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip
RUN apt install -y tesseract-ocr
RUN apt install -y libtesseract-dev
RUN apt-get install -y  tesseract-ocr-deu
WORKDIR /app
COPY /back/requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN cp /back/220207_model/* /usr/local/Cellar/tesseract/5.0.1/share/tessdata
CMD [ "chmod", "+x","./entrypoint.sh" ]