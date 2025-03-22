#!/bin/bash
pip install --upgrade pip
pip install --no-cache-dir --force-reinstall scikit-learn
pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt found, skipping..."
