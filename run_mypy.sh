#!/bin/bash

target_version=$(cat mypy_version.txt) 
current_version=$(mypy --version | cut -d " " -f2)

if [ "$target_version" != "$current_version" ]; then
    echo "Error: Exepected mypy==$target_version, found mypy==$current_version"
    exit 1
fi

output=$(mypy "$@" | grep --file mypy_files.txt)

if ! [ -z "$output" ]; then
    echo "$output"
    exit 1
fi

