# FROM docker.hobot.cc/dlp/base:centos7.6-extra-gcc5.4
FROM docker.hobot.cc/imagesys/hat:pilot-runtime-cu111-hdflow-20230506-torch1102-aidisdk0111-py38

SHELL [ "/bin/bash", "-l", "-c" ]

# install docker client
ADD dev/dockerfiles/docker-ce.repo /etc/yum.repos.d/docker-ce.repo
RUN yum install docker-ce -y

# config pip
ADD dev/dockerfiles/pip.conf /etc/pip.conf
RUN pip3 install --upgrade pip 

# Increase the default number of processes
RUN sed -i 's/4096/65535/g' /etc/security/limits.d/20-nproc.conf

# remove mxnet
RUN pip3 uninstall mxnet-horizon-cu111 -y

# update hatbc
RUN pip3 uninstall hatbc -y
RUN pip3 install -U hatbc -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

# install correct version hbdk 
RUN pip3 uninstall hbdk-internal -y
RUN pip3 uninstall hbdk -y
RUN pip3 install -U hbdk==3.39.2 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

# rm our pypi config
RUN rm -f /etc/pip.conf
