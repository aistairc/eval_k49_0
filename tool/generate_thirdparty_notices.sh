#!/bin/bash

DOCKER_IMAGE_NAME=ait_license_base:latest

echo start docker clean up...
docker rmi ${DOCKER_IMAGE_NAME}
docker system prune -f

echo start docker build...
docker build -f ../deploy/container/dockerfile -t ${DOCKER_IMAGE_NAME} ../deploy/container

LICENSE_DOCKER_IMAGE_NAME=ait_license_thirdparty_notices

echo start docker clean up...
docker rmi ${LICENSE_DOCKER_IMAGE_NAME}
docker system prune -f

echo start docker build...
docker build -t ${LICENSE_DOCKER_IMAGE_NAME} -f ../deploy/container/dockerfile_license ../deploy/container

echo run docker...
docker run ${LICENSE_DOCKER_IMAGE_NAME} >../ThirdPartyNotices.txt

read
