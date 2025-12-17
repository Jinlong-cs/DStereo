# docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-${COMMIT_ID:0:7}
# FROM docker.hobot.cc/imagesys/base:centos7.6-gcc5.4-py3.8-cuda11.1-horizon
FROM docker.hobot.cc/dlp/base:centos7.6-gcc5.4-py3.8-cuda11.1-horizon

SHELL [ "/bin/bash", "-l", "-c" ]

RUN rm /etc/yum.repos.d/nux-dextop.repo

# install docker client
ADD dev/dockerfiles/docker-ce.repo /etc/yum.repos.d/docker-ce.repo
RUN yum install docker-ce -y

# config pip
ADD dev/dockerfiles/pip.conf /etc/pip.conf
RUN pip3 install --upgrade pip

RUN pip3 uninstall enum34 tensorflow tensorflow-estimator setuptools -y
RUN pip3 install --upgrade pip setuptools==58.0.4 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

# Increase the default number of processes
RUN sed -i 's/4096/65535/g' /etc/security/limits.d/20-nproc.conf

# install pytorch
RUN pip3 install torch==1.10.2+cu111 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple && pip3 install torchvision==0.11.3+cu111 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple
RUN pip3 install torchaudio==0.10.2+cu111 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple

# install requirements
ADD requirements ./requirements
ADD requirements.txt .
RUN pip3 install --use-deprecated=legacy-resolver --no-cache-dir -r requirements.txt
RUN rm -drf requirements*
RUN pip3 install -U hbdk-internal -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
RUN pip3 install --no-cache-dir -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu111/torch1102 --trusted-host art-internal.hobot.cc
RUN pip3 install --no-cache-dir -U horizon-plugin-profiler -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
