# Layers Sustainability Analysis framework (LSA)
[[Presentation]](#)  [[Project]](https://github.com/khalooei/LSA) [[Paper]](https://arxiv.org/abs/2202.02626)

![image info](./imgs/LSA.jpg)

LSA stands for Layer Sustainability Analysis for the analysis of layer vulnerability in a given neural network. LSA can be a helpful toolkit to assess deep neural networks and to extend the adversarial training approaches towards improving the sustainability of model layers via layer monitoring and analysis. The LSA framework identifies a list of Most Vulnerable Layers (MVL list) of a given network. The relative error, as a comparison measure, is used to evaluate representation sustainability of each layer against adversarial attack inputs. 

[![License: MIT](https://img.shields.io/github/license/khalooei/LSA?&color=brightgreen)](https://github.com/khalooei/LSA/blob/master/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/v/layer-sustainability-analysis.svg?&color=orange)](https://pypi.org/project/layer-sustainability-analysis/)
[![Documentation Status](./imgs/bdg.svg)](#)
[![Release Status](https://img.shields.io/github/release/khalooei/LSA.svg?&color=blue)](https://github.com/khalooei/LSA/releases)

## Overview
Sustainability and vulnerability in different domains have many definitions. In our case, the focus is on certain vulnerabilities that fool deep learning models in the feed-forward propagation approach. One main concentration is therefore on the analysis of forwarding vulnerability effects of deep neural networks in the adversarial domain. Analyzing the vulnerabilities of deep neural networks helps better understand different behaviors in dealing with input perturbations in order to attain more robust and sustainable models.
The proposed approach for obtaining robust neural networks to fend off adversarial attacks is based on a layer-wise regularization (LR) over LSA proposal(s) for adversarial training (AT); i.e. the AT-LR procedure. AT-LR could be used with any benchmark adversarial attack to reduce the vulnerability of network layers and to improve conventional adversarial training approaches. The proposed idea performs well theoretically and experimentally for state-of-the-art multilayer perceptron and convolutional neural network architectures.

![image info](./imgs/LSA-proper-mini.gif)


## Getting Started

<details><summary>Easy installation</summary><p>

```python
 pip install layer-sustainability-analysis
```
</p></details>


<details><summary>Easy usage</summary><p>

```python
from layer-sustainability-analysis import LayerSustainabilityAnalysis as LSA
lsa = LSA(pretrained_model=model)
lst_comparison_measures = LSA.representation_comparison(img_clean=selected_clean_sample, img_perturbed=selected_pertubed_sample, measure ='relative-error')
```
</p></details>
