# Driving Scene VLM

[![CI](https://github.com/jatinmallarapu/DrivingSceneVisionLanguageModel/actions/workflows/ci.yml/badge.svg)](https://github.com/jatinmallarapu/DrivingSceneVisionLanguageModel/actions/workflows/ci.yml)

Fine-tune Qwen2.5-VL on driving-scene question answering and compare it against a zero-shot baseline using a scene-held-out split.

| Checkpoint | Multiple-choice accuracy | Open-answer ROUGE-L |
| --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | Pending GPU run | Pending GPU run |
| QLoRA, rank 16 | Pending GPU run | Pending GPU run |

Run `make test` after installing `service-requirements.txt` and pytest. Gateway tests cover empty and overlong questions, corrupted and blank frames, and preservation of the user-message boundary for adversarial text. The model is mocked in these tests.

This is an offline research evaluation. It produces text answers from images and has no vehicle control interface.

## Run the experiment

Use Linux, Python 3.11 and a CUDA GPU with BF16 support. Install with `python -m pip install -r requirements.txt`. Obtain DriveLM annotations and nuScenes images through their official distribution channels. This repository expects a normalized JSONL manifest rather than raw annotations. Each row contains `id`, `scene`, `image` (relative path), `question` and `answer`. Include a `choices` mapping and make the answer an option letter for multiple-choice rows. Map multi-camera rows to the image specified by the question or exclude them and report the excluded count.

```bash
python prepare.py data/examples.jsonl --image-root data/nuscenes
make train REVISION=MODEL_COMMIT_SHA
make eval REVISION=MODEL_COMMIT_SHA
make merge REVISION=MODEL_COMMIT_SHA
```

Use the same immutable base model revision and test manifest for both runs. The evaluation JSON retains per-example predictions for failure analysis. The split groups by scene to prevent adjacent frames leaking between partitions. Train and test accuracy must be reported separately for multiple-choice and open answers. Capture the GPU type, wall time and billing rate before quoting cost.

## Serve the merged model

On the Linux inference host, run `vllm serve models/merged --served-model-name driving-scene --host 127.0.0.1 --port 8001`. Build the gateway with `docker build -t driving-scene-vlm .`. Then run:

```bash
docker run --rm --network host -e VLLM_URL=http://127.0.0.1:8001/v1 driving-scene-vlm uvicorn service:app --host 127.0.0.1 --port 8000
curl -F image=@frame.jpg -F 'question=What is the pedestrian doing?' http://127.0.0.1:8000/predict
```

Remaining before reporting a model result: validate against the raw annotation distribution, run base and tuned evaluations, calibrate any judge against human-reviewed examples, and document image-use rights before publishing example frames or an adapter.

[DriveLM](https://github.com/OpenDriveLab/DriveLM) · [nuScenes terms](https://www.nuscenes.org/terms-of-use) · [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
