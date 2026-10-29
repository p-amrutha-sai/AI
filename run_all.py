"""Reproduce all numerical outputs with Python's standard library."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
commands = [
    ["-m","unittest","discover","-s","tests","-v"],
    ["-m","recsys","train","--episodes","200","--rounds","24","--seed","7"],
    ["-m","recsys","evaluate","--seeds","12","--seed-start","100","--rounds","24"],
    ["-m","recsys","demo","--seed","42","--rounds","24"],
]
for args in commands:
    subprocess.run([sys.executable,*args],cwd=root,check=True)
