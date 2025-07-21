#!/bin/bash

cd ./out && rm -r * 
cd ..
python drebin.py malwares_app/*.apk ./out/
mv ./out/results/*.json ./out/results/results.json
