###################
# Builder Stage
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
ARG DEBIAN_FRONTEND=noninteractive

# Don't write .pyc bytecode
# ENV PYTHONDONTWRITEBYTECODE=1

# Create workspace working directory
RUN mkdir /build
WORKDIR /build

# clean some disk space
RUN rm -rf /usr/share/dotnet
RUN rm -rf /opt/ghc
RUN rm -rf "/usr/local/share/boost"
RUN rm -rf "$AGENT_TOOLSDIRECTORY"

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
    pip install -U -I torch==2.1.0+cu121 torchvision==0.16.0+cu121 --extra-index-url "https://download.pytorch.org/whl/cu121" && \
    pip install -r requirements-build.txt && \
    pip install --no-deps xformers==0.0.22.post7

###################
# Runtime Stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as runtime

# Use bash shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash

# Python logs go strait to stdout/stderr w/o buffering
ENV PYTHONUNBUFFERED=1

# Don't write .pyc bytecode
# ENV PYTHONDONTWRITEBYTECODE=1

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
RUN wget https://github.com/runpod/runpodctl/releases/download/v1.14.3/runpodctl-linux-amd64 -O runpodctl && \
    chmod a+x runpodctl && \
    mv runpodctl /usr/local/bin

ENV VIRTUAL_ENV=/workspace/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
RUN echo "source ${VIRTUAL_ENV}/bin/activate" >> /root/.bashrc

# Workaround for:
#   https://github.com/TimDettmers/bitsandbytes/issues/62
#   https://github.com/TimDettmers/bitsandbytes/issues/73
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.1/targets/x86_64-linux/lib/"
#RUN ln /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12.1.55 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudart.so
#RUN ln /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.12.1.55 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libnvrtc.so.12.1.55

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
