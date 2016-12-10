TF_BINARY_URL = https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl

all: env/bin/python

env/bin/python:
	virtualenv env -p python2.7 --no-site-packages
	env/bin/pip install --upgrade pip
	env/bin/pip install wheel
	env/bin/pip install --upgrade $(TF_BINARY_URL) 
	env/bin/pip install -r requirements.txt

