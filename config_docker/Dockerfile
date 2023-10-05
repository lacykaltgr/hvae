FROM python

WORKDIR /app

#RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY entrypoint.sh app/
COPY start_jupyter.sh app/

RUN chmod +x /entrypoint.sh
RUN chmod +x /start_jupyter.sh

ENTRYPOINT ["/entrypoint.sh"]