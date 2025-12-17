# FRROM docker.hobot.cc/dlp/base:centos7.6-extra-gcc5.4
FROM docker.hobot.cc/imagesys/base:centos7.6-gcc5.4-py3.8-cuda10.2-release

SHELL [ "/bin/bash", "-l", "-c" ]

# install docker client
ADD dev/dockerfiles/docker-ce.repo /etc/yum.repos.d/docker-ce.repo
RUN yum install docker-ce -y

# config pip
ADD dev/dockerfiles/pip.conf /etc/pip.conf
RUN pip3 install --upgrade pip

# Increase the default number of processes
RUN sed -i 's/4096/65535/g' /etc/security/limits.d/20-nproc.conf

# install pytorch
RUN pip3 install torch==1.10.2+cu102 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple && pip3 install torchvision==0.11.3+cu102 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple
RUN pip3 install torchaudio==0.10.2+cu102 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple

# install requirements
ADD requirements ./requirements
ADD requirements.txt .
RUN pip3 install --use-deprecated=legacy-resolver --no-cache-dir -r requirements.txt
RUN rm -drf requirements*
RUN pip3 install -U hbdk-internal -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
RUN pip3 install --no-cache-dir -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu102/torch1102 --trusted-host art-internal.hobot.cc
RUN pip3 install --no-cache-dir -U horizon-plugin-profiler -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

# rm our pypi config
RUN rm -f /etc/pip.conf
