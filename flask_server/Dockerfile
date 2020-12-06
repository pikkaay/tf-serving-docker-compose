FROM python:3.7.1


ADD . /flask_server
WORKDIR /flask_server

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN chown -Rh $user:$user /flask_server
CMD [ "flask", "run" ]
