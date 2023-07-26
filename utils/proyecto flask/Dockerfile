FROM python:3.9
WORKDIR /app
COPY . /app
# Esto es de numpy por si acaso me va mal en la instalación
#RUN apt-get update && apt-get install -y build-essential libopenblas-dev liblapack-dev
# atento a este linea por si hay que cambiar algo
RUN python3.9 -m pip install -r requirements.txt
EXPOSE 80

CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]