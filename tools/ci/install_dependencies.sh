#!/bin/bash

echo "Installing dependencies"

set -eu

set -x

echo "OS_TYPE = $OS_TYPE" 

if [ "$OS_TYPE" = "ubuntu-latest" ]; then
    sudo apt update
    sudo apt install -y graphviz
elif  [ "$OS_TYPE" = "macos-latest" ]; then
    brew install graphviz
else
    echo "Unknown OS_TYPE: $OS_TYPE"	
fi

echo "Done installing dependencies"
