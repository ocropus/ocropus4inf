# OCRopus 4 Inferece

This package implements OCRopus 4 inference.

Right now, there is a small command line program, a library, and a Jupyter Notebook demo.

- `ocropus4` -- command line program for recognizing pages and lines
- `ocropus4inf` -- library implementing page recognition
- `pagerec.py` -- demo notebook

Comments:

- training and inference are two separate packages; training will be released separately
- the models are completely unoptimized, both in terms of runtime and in terms of error rates
- this uses loaded JIT models, which are slow the first few times around

Next Steps:

- optimize the models (memory, speed, error rate)
- transformer-based text line recognizer
- linearizer and/or bounding box extractor (using transformers)
- integrate deep-learning document cleanup