# GCARec
This is our Tensorflow implementation for the paper:

>Mengyuan Jing, Yanmin Zhu*, Tianzi Zang, Jiadi Yu, and Feilong Tang. Graph Contrastive Learning with Adaptive Augmentation for Recommendation. In ECML_PKDD'22, Grenoble, France, September 19-23, 2022.("*" = Corresponding author)

**Please cite our paper if you use our codes. Thanks!**

This project is based on [NeuRec](https://github.com/wubinzzu/NeuRec/) and [SGL](https://github.com/wujcan/SGL-TensorFlow). Thanks to the contributors.
## Environment Settings

- python == 3.6.15. 
- Tensorflow-gpu == 1.15.0
- numpy == 1.19.5
- scipy == 1.5.3
- cython == 0.29.23

## Quick Start
**Firstly**, compline the evaluator of cpp implementation:

```bash
python setup.py build_ext --inplace
```

If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.

**Note that the cpp implementation is much faster than python.**

Further details, please refer to [NeuRec](https://github.com/wubinzzu/NeuRec/)

**Secondly**, specify dataset and recommender in configuration file *NeuRec.properties*.

Specify hyperparameters in configuration file *./conf/GCARec.properties*.

**Finally**, run [main.py](./main.py) in IDE or with command line:

```bash
python main.py
```

