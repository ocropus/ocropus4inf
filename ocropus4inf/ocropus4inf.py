#!/usr/bin/env python

from typing import List
import ocropus4inf.ocrinf as inf
import imageio.v2 as imageio
import os
import json

import typer

from . import ocrinf as inf


app = typer.Typer()


textmodel = os.environ.get(
    "TEXT_MODEL",
    "https://storage.googleapis.com/ocro-models/v1/lstm_resnet_v2-038-000330009.jit",
)
segmodel = os.environ.get(
    "SEG_MODEL",
    "https://storage.googleapis.com/ocro-models/v1/seg_unet_v2-023-000272940.jit",
)
device = os.environ.get("DEVICE", "cuda:0")


@app.command()
def lines(files: List[str], nlbin: bool = False, verbose: bool = False):
    """Perform recognition on a set of lines and output the results as text."""
    textrec = inf.WordRecognizer(textmodel, device=device)
    images = [inf.autoinvert(imageio.imread(arg)/255.0) for arg in files]
    results = textrec.inference(images)
    for arg, result in zip(files, results):
        base = os.path.splitext(arg)[0]
        output_file = base + ".txt"
        with open(output_file, "w") as stream:
            stream.write(result)
        if verbose:
            print(f"{arg}: \"{result}\"")


@app.command()
def showpage(fname, nlbin: bool = False):
    """Perform recognition on a single page and display the result using Matplotlib."""
    import matplotlib.pyplot as plt
    pagerec = inf.PageRecognizer(textmodel=textmodel, segmodel=segmodel, device=device)
    image = imageio.imread(fname) / 255.0
    pagerec.recognize(image)
    pagerec.draw_overlaid()
    plt.show()


@app.command()
def pages2json(files: List[str], nlbin: bool = False):
    """Recognize pages and output results as JSON."""
    pagerec = inf.PageRecognizer(textmodel=textmodel, segmodel=segmodel, device=device)
    mode = "none" if not nlbin else "binarize"
    for arg in files:
        print("# processing", arg)
        image = imageio.imread(arg)
        if image.dtype == "uint8":
            image = image / 255.0
        result = pagerec.recognize(image)
        base = os.path.splitext(arg)[0]
        output_file = base + ".json"
        json.dump(result, open(output_file, "w"), indent=2)


if __name__ == "__main__":
    app()
