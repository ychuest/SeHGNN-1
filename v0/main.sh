#!/bin/bash

set -x

if [ ! -f main.py ]; then
    cd v0
fi

python main.py
