#!/bin/bash
mkdir /opt/ml/
mkdir /opt/ml/model
mkdir /opt/program/
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
git clone $GIT_REPO /opt/program
wget -O /opt/ml/model/model.pkl $MODEL_URL

echo $VAR1
ls /opt/program/
ls /opt/ml/model/
export FLASK_APP=/opt/program/serve.py
export FLASK_RUN_PORT=8080
flask run 

