#!/bin/bash
export PYTHONPATH=.: 
source ./deploy/secret_env_vars.sh
adev runserver website --root website --port 8001
