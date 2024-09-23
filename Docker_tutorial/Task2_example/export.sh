#!/usr/bin/env bash

# Stop at first error
set -e

# bash Docker_tutorial/Task2_example/build.sh

DOCKER_TAG="hntsmrgmid" # change this as needed

echo "=+= Exporting the Docker image to a tar.gz file"
docker save $DOCKER_TAG | gzip -c > ${DOCKER_TAG}.tar.gz

echo "=+= Docker image exported successfully to ${DOCKER_TAG}.tar.gz"