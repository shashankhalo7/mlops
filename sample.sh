#!/bin/bash
mkdir /opt/ml/
mkdir /opt/ml/model
mkdir /opt/program/
git clone $GIT_REPO /opt/program
wget -O /opt/ml/model/model.pkl $MODEL_URL

echo $VAR1
ls /opt/program/
ls /opt/ml/model/
python /opt/program/serve.py
