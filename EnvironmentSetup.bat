@echo off

mkdir python-3.11.9
cd python-3.11.9

::downloads embeded python 3.11.9
curl -L -o python-3.11.9-embed-amd64.zip https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
unzip python-3.11.9-embed-amd64.zip
del python-3.11.9-embed-amd64.zip

::this portion downloads and installs pip
curl -L https://bootstrap.pypa.io/get-pip.py -o get-pip.py
.\python.exe get-pip.py
del get-pip.py

::enabling import site so virtualenv can find packages
echo import site >> python311._pth

::installing virtualenv through pip
.\Scripts\pip.exe install virtualenv



::going back to project root folder
cd ..

::Create and Activate Virtual Environment
.\python-3.11.9\Scripts\virtualenv.exe -p .\python-3.11.9\python.exe BanglaLekha
.\BanglaLekha\Scripts\activate

::sanity check 
python --version
which python

