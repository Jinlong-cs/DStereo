FROM docker.hobot.cc/dlp/base:centos7.6-gcc5.4-py3.10-cpu-20231113
SHELL [ "/bin/bash", "-l", "-c" ]

RUN rm /etc/yum.repos.d/nux-dextop.repo

ENV PATH /usr/local/python3.10.12/bin:$PATH
ENV SSL_CERT_FILE /usr/local/python3.10.12/lib/python3.10/site-packages/certifi/cacert.pem
ENV LD_LIBRARY_PATH=/usr/lib64/mpich-3.2/lib:${LD_LIBRARY_PATH}

# install docker client
ADD dev/dockerfiles/docker-ce.repo /etc/yum.repos.d/docker-ce.repo
RUN yum install docker-ce -y

# config pip
ADD dev/dockerfiles/pip.conf /etc/pip.conf
RUN pip3 install --upgrade pip

RUN pip3 install wheel mpi4py -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

# Increase the default number of processes
RUN sed -i 's/4096/65535/g' /etc/security/limits.d/20-nproc.conf

# install pytorch
RUN pip3 install torch==2.0.1+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple
RUN pip3 install torchvision==0.15.2+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple
RUN pip3 install torchaudio==2.0.2+cpu -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple

# install requirements
ADD requirements ./requirements
ADD requirements.txt .
RUN pip3 install --use-deprecated=legacy-resolver --no-cache-dir -r requirements.txt
RUN rm -drf requirements*
RUN pip3 install -U hbdk-internal -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
RUN pip3 install --no-cache-dir -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cpu/torch201 --trusted-host art-internal.hobot.cc
RUN pip3 install --no-cache-dir -U horizon-plugin-profiler -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
