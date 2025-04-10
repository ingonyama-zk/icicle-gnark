# Start with CUDA base image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS cuda-base

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    git \
    cmake \
    curl \
    ca-certificates \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install Go
ENV GOLANG_VERSION=1.22.0
RUN curl -L https://go.dev/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz | tar -xz -C /usr/local
ENV PATH="/usr/local/go/bin:${PATH}"
RUN go version

WORKDIR ./

# Copy repository
COPY . .

# Explicitly fix go.mod versions
WORKDIR ./gnark-ffi/go

RUN go mod tidy

RUN ICICLE_DIR=$(go env GOMODCACHE)/github.com/ingonyama-zk/icicle-gnark/v3@v3.2.2; \
    cd $ICICLE_DIR/wrappers/golang; \
    /bin/bash build.sh -curve=bn254; \

ENV ICICLE_BACKEND_INSTALL_DIR=/usr/local/lib/backend

# Add steps below to run your application
