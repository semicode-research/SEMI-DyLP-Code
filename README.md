\# SEMI-DyLP



This repository provides the official implementation of \*\*SEMI-DyLP\*\* for continuous-time dynamic link prediction.



The corresponding manuscript is:



\*\*Event-Aware Representation Learning with Selective Multi-Step Influence Propagation for Dynamic Link Prediction\*\*



\## Overview



SEMI-DyLP is designed for dynamic link prediction in continuous-time interaction networks. The model performs event-wise representation updating and selectively propagates temporal influence through first- and second-order neighborhoods while preserving temporal causality.



The implementation includes the main components for:



\* Event-aware node representation updating

\* First- and second-order temporal influence propagation

\* Attention-based neighbor aggregation

\* Temporal decay modeling

\* Dynamic link prediction

\* Model training and evaluation



\## Repository Structure



```text

SEMI-DyLP/

├── code/

│   ├── attention.py

│   ├── combiner.py

│   ├── datasets.py

│   ├── decayer.py

│   ├── edge\_updater.py

│   ├── model.py

│   ├── node\_updater.py

│   ├── test.py

│   └── train.py

├── data/

├── .gitignore

├── README.md

└── requirements.txt

```



\## Datasets



The experiments in the manuscript are conducted on the following temporal interaction datasets:



\* Hypertext

\* Enron

\* UCI

\* Radoslaw

\* Email-Eu



The processed data files used by the implementation are provided in the `data/` directory.



\## Requirements



The implementation is based on Python and PyTorch.



The main dependencies are:



\* PyTorch 2.5.1

\* NumPy 2.1.1

\* SciPy 1.14.1

\* scikit-learn 1.6.1



Install the required packages using:



```bash

pip install -r requirements.txt

```



\## Training



The main training script is:



```text

code/train.py

```



The training procedure can be executed using the corresponding dataset and experimental settings described in the manuscript.



\## Testing



The main evaluation script is:



```text

code/test.py

```



The testing procedure evaluates the trained model for dynamic link prediction using the experimental settings reported in the manuscript.



\## Main Components



The implementation contains the following major modules:



\* `attention.py`: attention-based neighbor aggregation

\* `combiner.py`: representation combination module

\* `datasets.py`: temporal dataset loading and processing

\* `decayer.py`: temporal decay module

\* `edge\_updater.py`: edge representation updating

\* `model.py`: main SEMI-DyLP model

\* `node\_updater.py`: node state updating

\* `train.py`: model training

\* `test.py`: model evaluation



\## Reproducibility



This repository contains the implementation used for the experiments reported in the manuscript, including the model architecture, temporal updating modules, attention mechanism, data processing components, training procedure, and evaluation procedure.



The source code and experimental data are provided to facilitate reproducibility of the reported results.



\## Citation



If you find this work useful, please consider citing the corresponding paper.



Citation information will be updated after publication.



