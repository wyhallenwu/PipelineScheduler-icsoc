# RetinaFace in PyTorch

To convert retinaface with a `BatchedNMSPlugin` in TRT run 
```
python retina_batched_nms.py --device cpu
```

The majority of the code is burrowed from an [PyTorch](https://pytorch.org/) implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641) by [bigbug6](https://github.com/biubug6), available [here](https://github.com/biubug6/Face-Detector-1MB-with-landmark).
