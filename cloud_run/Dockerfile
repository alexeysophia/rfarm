# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        xz-utils \
        libx11-6 \
        libxi6 \
        libxxf86vm1 \
        libxfixes3 \
        libxrender1 \
        libgl1 \
        libglu1 \
        python3 \
        python3-pip \
        python3-venv \
        bash \
    && rm -rf /var/lib/apt/lists/*

ARG BLENDER_MAJOR_VERSION=4.5
ARG BLENDER_MINOR_VERSION=0
ARG BLENDER_VERSION=${BLENDER_MAJOR_VERSION}.${BLENDER_MINOR_VERSION}
ARG BLENDER_DIST=blender-${BLENDER_VERSION}-linux-x64

RUN wget -q https://download.blender.org/release/Blender${BLENDER_MAJOR_VERSION}/${BLENDER_DIST}.tar.xz -O /tmp/blender.tar.xz \
    && tar -xJf /tmp/blender.tar.xz -C /opt \
    && ln -s /opt/${BLENDER_DIST}/blender /usr/local/bin/blender \
    && rm /tmp/blender.tar.xz

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py render_worker.py start.sh ./
RUN chmod +x start.sh

ENV BLENDER_USER_CONFIG=/tmp/blender
ENV BLENDER_SYSTEM_SCRIPTS=/opt/${BLENDER_DIST}/${BLENDER_VERSION}/scripts
ENV BLENDER_SYSTEM_DATAFILES=/opt/${BLENDER_DIST}/${BLENDER_VERSION}/datafiles

CMD ["/app/start.sh"]
