#!/bin/bash
echo 'Running linters...'
pip install flake8 black
black src/
flake8 src/
