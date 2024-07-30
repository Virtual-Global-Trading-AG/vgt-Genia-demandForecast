#!/bin/bash

# Check if docker-compose is running
if sudo docker-compose ps -q; then
    echo "docker-compose is running. Stopping and removing containers..."
    sudo docker-compose down
fi

# Check if docker-compose file exists
if [ -f docker-compose.yml ]; then
    echo "docker-compose.yml exists. Deleting it..."
    rm docker-compose.yml
fi

# Rebuild the Docker image
echo "Building Docker image..."
sudo docker build -t demandforecastmanager .

# Bring up the docker-compose services
echo "Starting docker-compose..."
sudo docker-compose up -d