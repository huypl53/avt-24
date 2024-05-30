#!/usr/bin/sh
# dpkg -s python3-venv # check if package existed or not
# sudo apt install python3-venv

# Init env
# python3 -m venv ~/enhancing

# Activate env
source ~/enhancing/bin/activate

# install pip
# python -m ensurepip --upgrade

# install dependencies
pip install -r requirements.txt

# check if path was added to .bashrc
# export PATH="$PATH:$HOME/bin/"
