FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

LABEL maintainer="hanyongming@gmail.com"
LABEL description="Dockerfile for Parrot model training and experiments"

# sudo docker image build -t parrotexps:latest .
# Based on
# https://gist.githubusercontent.com/ceshine/77623d9972c2369bf0ffd40068792caf/raw/7b84275eea4f0dddb1b067db75213eab5625b46f/Dockerfile
# https://wandb.ai/wandb_fc/pytorch-image-models/reports/A-faster-way-to-get-working-and-up-to-date-conda-environments-using-fastchan---Vmlldzo2ODIzNzA
# Feel free to simplify if you want

ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=7331


# Basic utils and portaudio
RUN apt-get update && \
    apt-get install -y --no-install-recommends bc curl dc man git git-doc psmisc vim wget unzip bzip2 sudo build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \  
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && \ 
    apt-get install -y libportaudio2 libportaudiocpp0 portaudio19-dev libsndfile1-dev

# more utils
RUN apt-get install -y blktrace linux-tools-generic strace tcpdump fd-find && \
    curl -LO https://github.com/BurntSushi/ripgrep/releases/download/13.0.0/ripgrep_13.0.0_amd64.deb && \
    sudo dpkg -i ripgrep_13.0.0_amd64.deb    


# Install miniconda
# Used a fixed path to try to debug conda solver bug
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \   
    rm -rf /tmp/* && \
    conda update -n base -c defaults conda

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers 

USER $USERNAME


# Install mamba
RUN conda install -y mamba -c conda-forge

# Workdir and environment related things
COPY . /home/$USERNAME/parrot_exps
WORKDIR /home/$USERNAME/parrot_exps

## Update the base env with environment.yml
RUN mamba env update --file ./environment.yml 


# For interactive shell
RUN conda init bash
RUN echo "conda activate base" >> /home/$USERNAME/.bashrc

# See also https://pythonspeed.com/articles/activate-conda-dockerfile/
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN pip install -r pip_specific_requirements.txt

CMD pwd
