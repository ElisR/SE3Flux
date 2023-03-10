ARG IMAGE=nvidia/cuda:12.0.0-devel-ubuntu22.04
FROM $IMAGE

ARG JULIA_RELEASE=1.8
ARG JULIA_VERSION=1.8.5


# julia

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install --yes --no-install-recommends \
                    # basic stuff
                    curl ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -s -L https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_RELEASE}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz | \
    tar -C /usr/local -x -z --strip-components=1 -f -


# system-wide packages

COPY Project.toml Manifest.toml /usr/local/share/julia/environments/v${JULIA_RELEASE}/

RUN JULIA_DEPOT_PATH=/usr/local/share/julia \
    julia -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

# generate the device runtime library for all known and supported devices
RUN JULIA_DEPOT_PATH=/usr/local/share/julia \
    julia -e 'using CUDA; CUDA.precompile_runtime()' && \
    chmod 644 /usr/local/share/julia/compiled/v${JULIA_RELEASE}/GPUCompiler/*/*.bc
    # TODO: fix this in GPUCompiler.jl


# user environment

# we use a single folder, /data, as the user depot regardless of the actual user
# (i.e., to be compatible with `docker run --user`, which might not have a $HOME)

RUN mkdir -m 0777 /data
ENV JULIA_DEPOT_PATH=/data:/usr/local/share/julia

ENV JULIA_HISTORY=/data/logs/repl_history.jl

ENV JULIA_CUDA_USE_BINARYBUILDER=false

WORKDIR "/workspace"
COPY examples /workspace/examples

ENTRYPOINT ["/usr/local/bin/julia"]