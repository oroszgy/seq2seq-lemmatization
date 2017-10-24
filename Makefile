RSYNC_SERVER = deep:~/workspace/s2s_lemmatization                                                                                                                                                   
LOCAL_FOLDER = .
PARAM = -avz --exclude-from .rsyncignore

install:
	mkdir -p deps
	conda install pytorch torchvision cuda80 -c soumith
	cd deps/src && git clone --recursive https://github.com/pytorch/pytorch && cd .. && pip install -e 'git+git@github.com:pytorch/pytorch#egg=pytorch'
	cd deps && pip install -e 'git+git@github.com:pytorch/text@v0.2.0#egg=torchtext'
	cd deps && pip install -e 'git+git@github.com:IBM/pytorch-seq2seq#egg=pytorch-seq2seq'
	pip install -r requirements.txt

dump_requirements:
	conda list -e --no-pip > requirements-conda.txt
	pip freeze > requirements.txt
	conda env export -n s2s_lemmatization -f environment.yaml

data:
	@echo TODO

rsync:
	@echo 'run make push or pull'

push:
	@echo "Push will delete files doesn't exists in local"
	rsync $(PARAM) $(LOCAL_FOLDER) $(RSYNC_SERVER)/../

pull:
	@echo "Pull will delete files doesn't exists in remote server"
	rsync $(PARAM) $(RSYNC_SERVER) $(LOCAL_FOLDER)/../

.PHONY: install dump_requirements data rsync push pull
