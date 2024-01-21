FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app
COPY . /app/

RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN chmod +x config_docker/start_jupyter.sh
CMD ["config_docker/start_jupyter.sh"]
ENV PYTHONPATH="${PYTHONPATH}:/app"
