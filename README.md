# mnist-ocr-ensemble-v3
v3 of a multi-optimizer PyTorch OCR ensemble for handwritten digits. Optimizers: SOAP, AdamW, Muon (Lion, SGD, AdaHessian dropped for weaker real-world performance). Adds a case-classifying router (digit/uppercase/lowercase/unknown) ahead of planned letter-reading models. No post-processing — raw output only.
