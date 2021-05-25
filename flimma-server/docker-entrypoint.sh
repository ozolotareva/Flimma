#!/bin/bash

python3 manage.py migrate
gunicorn hyfed_server:application --bind 0.0.0.0:8000 --timeout 1200 --workers 1
