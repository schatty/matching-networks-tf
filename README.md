# Mathing Networks for One Shot Learning in TensorFlow 2.0

Repository provides implementation of _Matching Networks for One Shot Learning_ paper (https://arxiv.org/abs/1606.04080) in Tensorflow 2.0. Model has been tested on Omniglot and miniImagenet datasets.

<img width="1044" alt="Screenshot 2019-04-12 at 10 38 28 AM" src="https://user-images.githubusercontent.com/23639048/56020148-28af9900-5d0f-11e9-8a2d-c463ea46d3d5.png">

## Dependencies and installation

* Project has been tested on Ubuntu 18.04 with Python 3.6.8 and TensorFflow 2.0.0-alpha0
* The dependencies are Pillow and tqdm libraries, which are included in setup requirements
* Training and evaluating require `matchnet` lib. Run `python setup.py install` to install it
* To download Omniglot dataset run `bash data/download_omniglot.sh` from repository's root
* miniImagenet was taken from excellent project (https://github.com/renmengye/few-shot-ssl-public) and placed into data/mini-imagenet folder

## Repository Structure

The repository organized as follows. `matchnet` folder contains library with model and data-loading routines. `data` serves as a default directory for the datasets (change configs to specify different data path). `scripts` contains training and evaluation scripts. `tests` provides minimal tests for training. `resulst` folder contains description of training configurations and results as well as tranining log info.

## Training and evaluating

Configuration of training and evaluation procedures is specified by .config files (specify `data.datsaet_path` if dataset has different path). Default config files for Omniglot and miniImagenet are `omniglot.conf` and `miniimagenet` respectively (omniglot set as a default choice of scripts' arguments). Scripts `run_train.py` and `run_eval.py` runs prodcures while `setup_train.py` and `setup_eval.py` contain basic logic for model launching.
To run training procedure run the following commands from repository's root
* `python scripts/train/run_train.py --config scripts/omniglot.conf` for Omniglot
* `python scripts/train/run_train.py --config scripts/miniimagenet.conf` for miniImagent

To run evaluation procedure run the following commands from repository's root
* `python scripts/eval/run_eval.py --config scripts/omniglot.conf` for Omniglot
* `python scripts/eval/run_eval.py --config scripts/miniimagenet.conf` for miniImanet

Training procedure generates log file that can be found in `results/logs` directory after training will be finished. Name of the log file contains date and time and will be printed in `stdout` in the beginning.

## Tests

To run basic tests run following command from root directory (for now tests required GPU support)
* `python -m unittest tests/*`

## Results

Obtained results for Omniglot after 30 epochs with `train` (`val` part was not engaged yet)

| Environment                 | 5-way-1-shot     | 5-way-5-shot     | 20-way-1-shot    | 20-way-5-shot    |
|-----------------------------|------------------|------------------|------------------|------------------|
| Accuracy                    | 95.9%            | 99.0%            | 88.8%            | 96.1%            |

## Acknowledgements

* Thanks to Albert Berenguel Centeno (https://github.com/gitabcworld) for his PyTorch implementation which helped me to sort out tough parts of the training procedure.

## References

[1] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, Daan Wierstra _Matching Networks for One Shot Learning_ (https://arxiv.org/abs/1606.04080)

[2] Brenden M. Lake, Ruslan Salakhutdinov, Joshua B. Tenenbaum _The Omniglot Challenge: A 3-Year Progress Report_ (https://arxiv.org/abs/1902.03477)
