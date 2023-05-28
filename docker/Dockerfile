###################
# Builder Stage
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder
LABEL org.opencontainers.image.licenses="AGPL-1.0-only"
ARG DEBIAN_FRONTEND=noninteractive

# Don't write .pyc bytecode
ENV PYTHONDONTWRITEBYTECODE=1

# Create workspace working directory
RUN mkdir /build
WORKDIR /build

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt-get install -y \
        git wget build-essential \
        python3-venv python3-pip \
        gnupg ca-certificates \
    && update-ca-certificates

ENV VIRTUAL_ENV=/workspace/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ADD requirements-build.txt /build
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv ${VIRTUAL_ENV} && \
    pip install -U -I torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url "https://download.pytorch.org/whl/cu118" && \
    pip install -r requirements-build.txt && \
    pip install --no-deps xformers==0.0.18

###################
# Runtime Stage
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as runtime

# Use bash shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash

# Python logs go strait to stdout/stderr w/o buffering
ENV PYTHONUNBUFFERED=1

# Don't write .pyc bytecode
ENV PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt install -y --no-install-recommends \
        wget bash curl git git-lfs vim tmux \
        build-essential lsb-release \
        python3-pip python3-venv \
        openssh-server \
        gnupg ca-certificates && \
    update-ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Install runpodctl
RUN wget https://github.com/runpod/runpodctl/releases/download/v1.9.0/runpodctl-linux-amd -O runpodctl && \
    chmod a+x runpodctl && \
    mv runpodctl /usr/local/bin

ENV VIRTUAL_ENV=/workspace/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
RUN echo "source ${VIRTUAL_ENV}/bin/activate" >> /root/.bashrc

# Workaround for:
#   https://github.com/TimDettmers/bitsandbytes/issues/62
#   https://github.com/TimDettmers/bitsandbytes/issues/73
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/targets/x86_64-linux/lib/"
RUN ln /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.11.8.89 /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so
RUN ln /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.11.8.89 /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc.so

# Vast.ai SSH ignores ENV vars unless fully exported
# Exporting anything with an _ should cover the bases
RUN env | grep _ >> /etc/environment;

ADD requirements-runtime.txt /
RUN pip install --no-cache-dir -r requirements-runtime.txt

WORKDIR /workspace
ARG CACHEBUST=1
RUN git clone https://github.com/victorchall/EveryDream2trainer

WORKDIR /workspace/EveryDream2trainer

ADD welcome.txt /
ADD start.sh /
RUN chmod +x /start.sh
CMD [ "/start.sh" ]
