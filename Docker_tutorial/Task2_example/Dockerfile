FROM --platform=linux/amd64 pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
# FROM --platform=linux/amd64 pytorch/pytorch
# FROM nvcr.io/nvidia/pytorch:22.12-py3


# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# COPY --chown=user:user requirements.txt /opt/app/
# COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user . /opt/app/

ENV nnUNet_results = "/opt/app/resources/nnUNet_results"

# RUN python -m pip install --upgrade pip
# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

RUN pip install \
    --user \
    -e /opt/app/resources/nnUNet-master

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]

