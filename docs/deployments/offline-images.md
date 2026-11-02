# Offline Container Images

This page shows how to package a server whose models use `env` requirements
into a container image that runs without network access. Network access is
needed only while the image is built.

## Bundle, Image, and Container

- **Bundle**: the directory `nomad export` writes. It holds the rewritten
  `nomad.yml`, the model files under `models/`, and one virtual environment per
  distinct `env` under `venv/`.
- **Image**: a read-only snapshot of a complete filesystem (operating system,
  Python, Nomad, and the bundle). You build it once and copy it to other
  machines.
- **Container**: a running instance of an image. `nomad serve` and every model
  subprocess run inside it, isolated from the host's files and Python.

## Why Export Runs Inside the Image Build

A virtual environment only works with the Python that built it. It links to
that interpreter's path, its scripts contain absolute paths, and its compiled
packages target one operating system and CPU architecture. Nomad also finds a
bundled environment by a checksum of its requirements, the Python version and
platform, and the Nomad installation.

As a result, a bundle exported on a laptop does not work when copied into an
image. Instead, run `nomad export` as a step of the image build. The
environments are then built by the image's own Python, at the image's own
paths, and the finished image carries everything they depend on.

## Workflow

```text
networked build machine                          offline target machine
(same CPU architecture as the target)
  docker build                                     docker run (no network)
    install Nomad                                    nomad serve
    copy config, models, packages      move            └─ model subprocess
    nomad export  ← downloads packages ─────image────►     uses the prebuilt venv
```

1. On a machine with network access and the same CPU architecture as the
   target, build the image. The `nomad export` step downloads and installs
   every `env` requirement.
2. Move the image to the target: push it to a registry the target can reach,
   or save it to a file (`docker save -o nomad.tar my-server:latest`), copy
   it, and load it there (`docker load -i nomad.tar`). On HPC systems without
   Docker, convert it to an Apptainer image
   (`apptainer build nomad.sif docker-archive://nomad.tar`) and copy the
   `.sif` file.
3. On the target, start a container from the image. Clients reach the server
   through the published port; nothing else needs the network.

## Example Dockerfile

```dockerfile
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
ENV UV_LINK_MODE=copy UV_SYSTEM_CERTS=1 NOMAD_CACHE=/opt/nomad

# Install Nomad from a source checkout so each model environment installs this
# same Nomad instead of a released version from the package index.
COPY nomad /src/nomad
RUN --mount=type=secret,id=nomad_build_ca,target=/etc/ssl/certs/ca-certificates.crt,required=false \
    uv pip install --system -e /src/nomad

# Local packages that env: entries reference, for example
#   env: my-model-package @ file:///src/my-model-package
COPY my-model-package /src/my-model-package
COPY server /src/server

RUN --mount=type=secret,id=nomad_build_ca,target=/etc/ssl/certs/ca-certificates.crt,required=false \
    nomad export /src/server/nomad.yml /opt/nomad

EXPOSE 8000
CMD ["nomad", "serve", "/opt/nomad/nomad.yml", \
     "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
```

`env` direct references must use paths that exist inside the image. Either
write them as image paths, as above, or use a placeholder such as
`file://${SRC}/my-model-package` and set `SRC` in the build.
`NOMAD_CACHE` must point at the export directory when the server runs, so that
Nomad finds the bundled environments.

## Building Behind a TLS-Intercepting Proxy

Some networks route HTTPS through a proxy that re-signs traffic with its own
root certificate. Downloads then fail with errors such as
`certificate signed by unknown authority` or `CERTIFICATE_VERIFY_FAILED` until
two separate places trust that root:

1. **The container engine**, which pulls base images such as
   `python:3.13-slim`. Add the root to the engine host's trust store. On macOS
   and Windows, the engine runs in a Linux virtual machine with its own trust
   store, so add the root inside that machine and restart the engine there.
2. **The build steps**, which download Python packages. The example
   Dockerfile mounts an optional build secret named `nomad_build_ca` over the
   image's system CA bundle for the two steps that download, and
   `UV_SYSTEM_CERTS=1` makes `uv` use that bundle. Pass a complete replacement
   bundle, with the public roots plus the proxy root:

   ```shell
   docker buildx build \
     --secret id=nomad_build_ca,src=/path/to/ca-bundle.pem \
     --tag my-server:latest .
   ```

The secret is only visible while those steps run and is not stored in the
image. Offline containers make no outbound connections, so the finished image
does not need the proxy root. To trust extra roots at run time anyway, see
[Passing Custom CA Certificates to the Image](guide.md#passing-custom-ca-certificates-to-the-image).

## Running the Image

On the target machine, publish the server's port so clients can reach it at
`http://<host>:8000/mcp`:

```shell
docker run -d --name my-server -p 8000:8000 my-server:latest
```

With Apptainer, run `apptainer run nomad.sif`. Apptainer shares the host's
network, so the server listens on port 8000 of the host directly.

To check on a networked machine that the server needs no network access, run
it on an internal Docker network, which has no route to the outside. Don't use
`--network none` for this check: it also disables published ports, so no
client can reach the server. Start the server, then query it from a second
container on the same network, which reaches the server by its container name:

```shell
docker network create --internal offline-check
docker run -d --name my-server --network offline-check my-server:latest
docker run --rm --network offline-check my-server:latest python -c "
import asyncio
from fastmcp import Client

async def main():
    async with Client('http://my-server:8000/mcp') as client:
        print([tool.name for tool in await client.list_tools()])

asyncio.run(main())
"
```

Give the server a few seconds to start before you query it; `docker logs
my-server` shows when it is ready. When you are done, remove the container and
the network with `docker rm -f my-server` and
`docker network rm offline-check`.

For GPUs, the host provides the NVIDIA driver and the image provides CUDA
through the model's PyTorch wheels. Pass `--gpus all` to Docker or `--nv` to
Apptainer.

## Run-Time Environment Builds Are Disabled

`nomad export` writes `tool_manager.build_envs: false` into the exported
`nomad.yml`. If a model's environment is missing when the server starts, that
model fails to load with an error naming the expected path, rather than trying
to download packages. The usual causes are that `NOMAD_CACHE` doesn't point at
the export directory, or that the server runs with a different Python or Nomad
installation than the one that ran `nomad export`.
