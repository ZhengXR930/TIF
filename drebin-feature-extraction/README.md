# Drebin Feature Extractor

Drebin static analysis based on Mobile Sandbox implementation by Michael Spreitzenbarth (research@spreitzenbarth.de).

This was later modified by Roberto Jordaney for Transcend (USENIX Security 2017). Later adapted by Feargus Pendlebury and Fabio Pierazzi for TESSERACT (USENIX Security 2017). Later further adapted by Feargus Pendlebury and Jacopo Cortellazzi for Prisma (S&P20). And than later updated in 2022 by Daniel Arp during his visit to the lab. 

## Installation

Create Python3 virtual environment (recommended)
```bash
python3 -m venv drebin-venv
source drebin-venv/bin/activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Install [GNUParallel](https://www.gnu.org/software/parallel/)

## Getting Started

Download sample APK files with
```bash
make download
```

Extract features with
```bash
make extract
```

Extract features in alternative JSON format that can be processed by [jq](https://stedolan.github.io/jq/)
```bash
make alt-extract
< data/processed/features.json | jq -r '.'
```



