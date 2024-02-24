
## Controlling Travel Path of Original Cobra


![Python 3.9.7](https://img.shields.io/badge/python-3.9.7-blue.svg)


Implementation of the paper : [Controlling Travel Path of Original Cobra](https://arxiv.org/abs/2210.10655)

## Abstract

In this paper we propose a kernel based COBRA which is a direct approximation of the original COBRA. We propose a novel tuning procedure for original COBRA parameters based on this kernel approximation. We show that our proposed algorithm provides much better accuracy than other COBRAs and faster than usual Gridsearch COBRA. We use two datasets to illustrate our proposed methodology over existing COBRAs.

## Split of the dataset

| Dataflow | Pretrain Dataset | Train Set (or Proximity Set) |
| --- | --- | --- |
| Original COBRA | $D_{k}$ | $D_{l}$ |
| Split Full Prox | $D_{k}$ | $D_{k} \cup D_{l}$ |
| No Split | $D_{k} \cup D_{l}$ | $D_{k} \cup D_{l}$ |


## Search Methods 

| Search Method | Description |
| --- | --- |
| Grid Search | Grid Search over the 100 grid points between the minimum and maximum prediction of Proximity Set Predictions |
| Random Search | Random Search over the 100 random points between the minimum and maximum prediction of Proximity Set Predictions |
| Gradient Descent | Gradient Descent over the 100 iterations both in proposed and Gradient COBRA |

## Datasets

The dataset taken from UCI regression dataset.




| Dataset                   | Observations | Dimensions |
|---------------------------|--------------|------------|
| airfoil                   | 1503         | 5          |
| autompg                   | 398          | 7          |
| breastcancer              | 198          | 33         |
| california housing dataset| 20640        | 8          |
| concreteslump             | 103          | 7          |
| energy                    | 768          | 8          |
| forest                    | 517          | 12         |
| servo                     | 167          | 4          |
| skillcraft                | 3395         | 19         |
| sml                       | 4137         | 26         |
| yacht                     | 308          | 6          |




## File Description

| File | Description |
| --- | --- |
| `final.py` | Run the base models, store the results, and distances for all dataset except California Housing and Boston Housing |
| `final_other.py` | Run the base models, store the results, and distances for California Housing and Boston Housing |
| `final2.py` | Run the proposed models, and othee COBRA search methods, store the results, and distances for all dataset except California Housing and Boston Housing |
| `final2_other.py` | Run the proposed models, and othee COBRA search methods, store the results, and distances for California Housing and Boston Housing |
| `final2_randomized.py` | Run the randomized search methods, store the results, and distances for all dataset except California Housing and Boston Housing |
| `final2_other_randomized.py` | Run the randomized search methods, store the results, and distances for California Housing and Boston Housing |

## Attribtion

1. Gradient Cobra is taken from [Gradient Cobra](https://github.com/hassothea/gradientcobra)
2. For base models [SciKit Learn](https://scikit-learn.org/stable/index.html) is used.
