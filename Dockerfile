# Utilizing Python 3.10 image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# Copying the requirements text file inside the docker container
COPY requirements.txt .

# Informing to install all the packages specified in the requirements text
RUN pip3 install -r requirements.txt

# Copying the app directory inside the docker container
COPY ./app /app

# Copying the models directory inside the docker container
COPY ./models /models

# Executing the command gunicorn with multiple workers to launch FastAPI
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:regressor_app"]