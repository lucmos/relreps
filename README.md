# Relative representations enable zero-shot latent space communication

[Slides](https://lucmos.github.io/relreps-presentation/) |
[OpenReview](https://openreview.net/forum?id=SrC-nwieGJ) |
[arXiv](https://arxiv.org/abs/2209.15430) |
[BibTeX](#bibtex)

<p align="center">
    <img alt="NN Template" src="./data/assets/teaser.gif">
</p>

[Luca Moschella](https://luca.moschella.dev/)\*,
[Valentino Maiorca](https://gladia.di.uniroma1.it/authors/maiorca/)\*,
[Marco Fumero](https://gladia.di.uniroma1.it/authors/fumero/),
[Antonio Norelli](https://noranta4.com/),
[Francesco Locatello](https://www.francescolocatello.com/),
[Emanuele Rodolà](https://gladia.di.uniroma1.it/authors/rodola/)

\* *equal contribution*

## Installation
<p align="left">
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.2.1-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

```bash
pip install git+ssh://git@github.com/lucmos/relreps.git
```


## Quickstart

### Development installation

Setup the development environment:

```bash
git clone git@github.com:lucmos/relreps.git
cd relreps
conda env create -f env.yaml
conda activate relreps
pre-commit install
dvc pull
```

> Refer to the [template documentation](https://grok-ai.github.io/nn-template/0.2/) for an high level overview of the code structure.

### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```


## BibTeX

```bibtex
@inproceedings{
    moschella2023relative,
    title={Relative representations enable zero-shot latent space communication},
    author={Luca Moschella and Valentino Maiorca and Marco Fumero and Antonio Norelli and Francesco Locatello and Emanuele Rodol{\`a}},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=SrC-nwieGJ}
}
```
