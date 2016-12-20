TF_BINARY_URL = https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl

all: env/bin/python

env/bin/python:
	virtualenv env -p /usr/bin/python2.7 --no-site-packages
	env/bin/pip install --upgrade pip
	env/bin/pip install wheel
	#env/bin/pip install --upgrade $(TF_BINARY_URL)
	env/bin/pip install -r requirements.txt
	git clone git@github.com:Rene90/gym-textgame env/src/gym-textgame
	git clone git@github.com:fchollet/keras env/src/keras
	env/bin/pip install -e env/src/gym-textgame
	env/bin/pip install -e env/src/keras

clean:
	rm -rfv bin develop-eggs dist downloads eggs env parts
	rm -fv .DS_Store .coverage .installed.cfg bootstrap.py
	rm -fv logs/*.txt
	find . -name '*.pyc' -exec rm -fv {} \;
	find . -name '*.pyo' -exec rm -fv {} \;
	find . -depth -name '*.egg-info' -exec rm -rfv {} \;
	find . -depth -name '__pycache__' -exec rm -rfv {} \;

.PHONY: clean
