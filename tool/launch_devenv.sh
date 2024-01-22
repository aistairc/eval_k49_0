#!/bin/bash

YML=docker/docker-compose.yml

echo Launching docker containers.

cp ../deploy/container/resolve-dependencies.sh docker/jupyter/resolve-dependencies.sh

docker-compose -f ${YML} up -d

echo Please wait until all docker containers are running ...

sleep 3

echo Opening Jupyter environment.
open http://localhost:9888?token=token

# read -p "Do you want to stop AIT development environment (containers)? <Y/N>" YN
# case ${YN} in
#   [yY]*)
#     docker-compose -f ${YML} down;
#   *)
#     read;
# esac
