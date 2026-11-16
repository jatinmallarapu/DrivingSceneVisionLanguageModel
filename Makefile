train:
	python experiment.py train --revision "$(REVISION)" --manifest data/splits/train.jsonl --epochs 1 --output runs/adapter

eval:
	python experiment.py eval --revision "$(REVISION)" --manifest data/splits/test.jsonl --output runs/base.json
	python experiment.py eval --revision "$(REVISION)" --manifest data/splits/test.jsonl --adapter runs/adapter --output runs/finetuned.json

merge:
	python experiment.py merge --revision "$(REVISION)" --adapter runs/adapter --output models/merged

test:
	python -m pytest -q
