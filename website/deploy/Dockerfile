FROM ubuntu:18.04

# Install Nginx.
RUN \
  apt-get update && \
  apt-get install -y nginx && \
  apt-get install -y vim && \
  apt-get install -y supervisor && \
  apt-get install -y software-properties-common && \
  apt-get install -y curl


RUN \
  add-apt-repository -y ppa:deadsnakes/ppa && \
  apt-get install -y python3.7 
  
RUN \
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
  python3.7 get-pip.py

COPY ./deploy/nginx.conf /etc/nginx/nginx.conf
COPY ./deploy/supervisord.conf /etc/supervisor/supervisord.conf

WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Expose ports.
EXPOSE 8080

ENTRYPOINT ["supervisord"]
# ENTRYPOINT ["/bin/bash"]
