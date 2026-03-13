#!/usr/bin/env python3

import os
import signal
import subprocess

signal.signal(signal.SIGINT, lambda *_: (print("\nBenchmark Interrupted."), os._exit(1)))

INDEX = ["0,0", "1,0", "2,0"]
PROMPTS = open("benchmark.prompts").read().splitlines()

for i in INDEX:
    print(f"INDEX: {i}")
    os.environ["INDEX"] = i
    subprocess.run(f'/app/app.py cat_mini.png 3', shell=True)

