"""Isolated Julia worker: deliberately imports no torch module."""
import argparse
from pathlib import Path
import numpy as np
from .io import read_json
from .pysr_baseline import _fit_pysr_in_process

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    args=parser.parse_args();settings=read_json(args.directory/"request.json")
    with np.load(args.directory/"request.npz") as data:
        _fit_pysr_in_process(data["features"],data["probabilities"],settings["names"],args.directory,
                             iterations=settings["iterations"],seed=settings["seed"])
    print("isolated PySR export verified",flush=True)

if __name__=="__main__":main()
