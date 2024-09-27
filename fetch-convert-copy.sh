#!/usr/bin/env bash

clear

./fetch-data.py
rm -rf ./lean_data
./data-to-lean.py
cp -vr ./lean_data/* ../lean/data/

echo -e "\e[1;32mDone!\e[0m\n\n"

