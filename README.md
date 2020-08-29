# Imbalanced Gradients: A New Cause of Overestimated Adversarial Robustness

This is the code for the paper "Imbalanced Gradients: A New Cause of Overestimated Adversarial Robustness"

## Prerequisites

* python (3.6.5)
* pytorch (1.1.0)
* torchvision (0.3.0)
*  numpy
* argparse
* tqdm

## Overview of the Code

```attack.py``` : The implementation of MD, MDMT and PGD attacks.

```eval.py``` :  An adversarial robustness evaluation script. **Edit it to load your own model for evaluation.** 

```models\``` : The directory for  models' codes.



### Running the code

* Evaluate under MD attack by running:
```bash
python eval.py --gpus 0 --md
```

* Evaluate under MDMT attack by running:
```bash
python eval.py --gpus 0 --mdmt
```





