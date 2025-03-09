# Hyperspherical Prototype Networks
This repository contains the PyTorch code for the NeurIPS 2019 paper "Hyperspherical Prototype Networks".
<br>
The paper is available here: https://arxiv.org/abs/1901.10514
<br><br>
<img src="data/spheres-visualization.png" alt="Drawing" style="width: 400px;"/>
<br><br>
The repository includes:
* Download link for pre-computed prototypes.
* Classification scripts for CIFAR-100, ImageNet-200, and CUB Birds.
* Script to construct your own prototypes.
* Joint classification and regression script for OmniArt.
* **NEW**: Einstein-Rosen bridge implementation that travels through the center of the poles of the hypersphere.

## Downloading and constructing hyperspherical prototypes

To obtain prototypes pre-computed for the paper, perform the following steps:
```
cd prototypes/
wget -r -nH --cut-dirs=3 --no-parent --reject="index.html*" http://isis-data.science.uva.nl/mettes/hpn/prototypes/
cd ..
```
The folder 'sgd' denotes the prototypes without semantic priors, 'sgd-sem' with semantic priors. The folders 'sem' and 'simplex' denote the baseline prototypes of Table 1.
<br><br>
To create your own prototypes, use the prototypes.py script. An example run for 100 classes and 50 dimensions:
```
python prototypes.py -c 100 -d 50 -r prototypes/sgd/
```
In case you want to construct prototypes on CIFAR-100 or ImageNet-200 with word2vec representations, please download the wtv files as follows:
```
mkdir -p wtv
cd wtv/
wget -r -nH --cut-dirs=3 --no-parent --reject="index.html*" http://isis-data.science.uva.nl/mettes/hpn/wtv/
cd ..
```

## Running hyperspherical prototype networks

To perform classification and joint optimization with Hyperspherical Prototype Networks, use the scripts that start with 'hpn_'.
<br>
For CIFAR-100 using 50-dimensional prototypes without semantic priors (akin to column 4 of Table 1 of the paper), run the following:
```
python hpn_cifar.py --datadir data/ --resdir res/ --hpnfile prototypes/sgd/prototypes-50d-100c.npy --seed 100
```
All the other scripts work precisely the same.
<br><br>
The CUB Birds dataset can be obtained from the original dataset: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
<br>
The prepared ImageNet-200 and OmniArt datasets can be obtained as follows:
```
cd data/
wget -r -nH --cut-dirs=3 --no-parent --reject="index.html*" http://isis-data.science.uva.nl/mettes/hpn/data/imagenet200/
wget -r -nH --cut-dirs=3 --no-parent --reject="index.html*" http://isis-data.science.uva.nl/mettes/hpn/data/omniart/
cd ..
```

## Einstein-Rosen Bridge for Hyperspherical Prototype Networks

This repository now includes an implementation of an Einstein-Rosen bridge that travels through the center of the poles of the hypersphere. The bridge creates a wormhole-like connection between antipodal points on the hypersphere, allowing for more efficient information transfer and potentially improved classification performance.

To run the HPN with Einstein-Rosen bridge on CIFAR-100:
```
python hpn_einstein_rosen.py --datadir data/ --resdir res/ --hpnfile prototypes/sgd/prototypes-50d-100c.npy --bridge_radius 0.3 --seed 100
```

For 3D embeddings, you can visualize the Einstein-Rosen bridge by adding the `--visualize` flag:
```
python hpn_einstein_rosen.py --datadir data/ --resdir res/ --hpnfile prototypes/sgd/prototypes-3d-100c.npy --bridge_radius 0.3 --visualize --seed 100
```

The visualization will show:
- Original points on the hypersphere (blue)
- Transformed points after passing through the bridge (red)
- North and South poles of the hypersphere (green and yellow)
- The bridge axis connecting the poles through the center (black line)

The bridge radius parameter controls how much of the hypersphere near the poles is affected by the wormhole. A smaller radius means a more localized effect.

Please cite the paper accordingly:
```
@inproceedings{mettes2019hyperspherical,
  title={Hyperspherical Prototype Networks},
  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
