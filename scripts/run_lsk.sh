#!/bin/bash

cd $(dirname $(dirname $0) )
conda activate openmmlab
python cli_lsk.py
