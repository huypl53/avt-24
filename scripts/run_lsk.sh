#!/bin/bash

cd $(dirname $(dirname $0) )
conda activate openmmlab
mkdir -p ./tmp/
python cli_lsk.py
