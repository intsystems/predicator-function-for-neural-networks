FROM nvidia/cuda:12.4.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/*.list

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget

RUN apt-get update && apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-distutils \
        curl \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN ln -s /usr/bin/python3.10 /usr/bin/python3

RUN mkdir /app
WORKDIR /app

RUN python -m pip install --upgrade pip && python -m pip install filelock

RUN wget https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp310-cp310-linux_x86_64.whl#sha256=9dde30f399ca22137455cca4d47140dfb7f4176e2d16a9729fc044eebfadb13a && \
    python -m pip install torch-2.5.1+cu124-cp310-cp310-linux_x86_64.whl

RUN wget https://download.pytorch.org/whl/cu124/torchvision-0.20.1%2Bcu124-cp310-cp310-linux_x86_64.whl#sha256=3a055e4e9040b129878d57c39db55f117f975899ff30dd70c8f2621d91170dbe && \
    python -m pip install torchvision-0.20.1+cu124-cp310-cp310-linux_x86_64.whl

RUN wget https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_scatter-2.1.2%2Bpt25cu124-cp310-cp310-linux_x86_64.whl && \
    pip install torch_scatter-2.1.2+pt25cu124-cp310-cp310-linux_x86_64.whl

# RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
# RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
# RUN wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl \
#     && pip install torch_scatter-2.1.0+pt112cu113-cp38-cp38-linux_x86_64.whl

COPY code/requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

