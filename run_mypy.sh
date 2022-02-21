#!/bin/bash

echo "Checking code with $(mypy --version)..."

output=$(mypy "$@" | grep --file mypy_files.txt)

if ! [ -z "$output" ]; then
    echo "$output"
    exit 1
fi

echo 'All good!'
