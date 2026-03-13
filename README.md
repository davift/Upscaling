# upscaling

Upscaling is a very specialized category of models that usually can only do one specific ratio (2x/4x/8x).

**Disclaimer:** The code in this repository is capable of downloading and running multiple *uncensored models. Use with resposibility and respect!

Content of the readme.
- Container
  - Build, Run, and Manage
- Applications/Scripts
  - Benchmarking
    - Runs the same batch of prompts against all models for comparison.
  - CLI
    - Can generates muiltiple images fromt he same prompt,
    - Easy to integrate with other apps or automations,
    - Has a randomizer function for parameters (strength and guidance).
  - Web-UI
    - (incomplete)

## Container

This container image includes all the required libraries and dependencies to run the models and scripts without any hassle. It is build based on **PyTorch v2.1.2** for compatibility with **NVIDIA Tesla P4** (Pascal architecture).

Build
```bash
docker build -t upscaling:v1.0 .
```

Run in Background
```bash
docker run -itd --gpus all -v $(pwd):/app -v $(pwd)/../models:/models -p 7860:7860 --name Upscaling upscaling:v1.0
```

Run in Background with Web-UI
```bash
docker run --rm -itd --gpus all -v $(pwd):/app -v $(pwd)/../models:/models -p 7860:7860 -e INDEX=0,0 --name Upscaling upscaling:v1.0 /app/app.py
```

Managing
```bash
docker exec -it Upscaling bash
docker logs -f Upscaling
docker stop Upscaling
docker rm Upscaling
```

## Application


### Benchmarking

Running
```bash
./benchmark.py
```

### CLI

Usage
```bash
INDEX=M,N python app.py input_file [prompt] [num_images] [num_saved_steps]
```
Note: INDEX refers to the model to be used, see code.

Examples
```bash
./app.py girl.png
./app.py car.png 3
INDEX=0,0 ./app.py car.png 0
```

### Web-UI

```bash
./app.py
```
Navigate to http://IP:7860/

## Avoid Duplication

This will rename all files to its MD5 hash to prevent duplication and tampering.

```bash
for f in *.png; do [ -f "$f" ] && h=$(md5sum "$f" | cut -d ' ' -f1) && mv -n -- "$f" "$h.png"; done
```
