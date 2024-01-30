
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


| Dataset | # of Features | # of Instances |
| --- | --- | --- |
| Challenger | 4 | 23 |
| Concrete Slump | 7 | 103 |
| Energy | 8 | 768 |
| Stock | 11 | 536 |
| Yacht | 6 | 308 |
| Autos | 25 | 159 |
| Breast Cancer | 33 | 194 |
| Fertility | 9 | 100 |
| Machine | 7 | 209 |
| Pendulum | 9 | 630 |
| Servo | 4 | 167 |
| California Housing | 8 | 20640 |
| Boston Housing | 13 | 506 |


## File Description

| File | Description |
| --- | --- |
| `final.py` | Run the base models, store the results, and distances for all dataset except California Housing and Boston Housing |
| `final_other.py` | Run the base models, store the results, and distances for California Housing and Boston Housing |
| `final2.py` | Run the proposed models, and othee COBRA search methods, store the results, and distances for all dataset except California Housing and Boston Housing |
| `final2_other.py` | Run the proposed models, and othee COBRA search methods, store the results, and distances for California Housing and Boston Housing |
| `final2_randomized.py` | Run the randomized search methods, store the results, and distances for all dataset except California Housing and Boston Housing |
| `final2_other_randomized.py` | Run the randomized search methods, store the results, and distances for California Housing and Boston Housing |

## Time Taken in search of $$\epsilon$$ search

\begin{table}[htbp]
\centering
\caption{Time in search of optimal hyperparameter for all datasets and methods The best three results are highlighted in green, yellow and red respectively.}
\begin{tabular}{lrrrrrrrrrrrr}
  \toprule
  \multirow{2}{*}{\textbf{Dataset}} & \multicolumn{4}{c}{\textbf{Original}} & \multicolumn{4}{c}{\textbf{Split Full Prox}} & \multicolumn{4}{c}{\textbf{No Split}} \\
  \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}
  & \textbf{Grid} & \textbf{Gradient} & \textbf{Random} & \textbf{Proposed} & \textbf{Grid} & \textbf{Gradient} & \textbf{Random} & \textbf{Proposed} & \textbf{Grid} & \textbf{Gradient} & \textbf{Random} & \textbf{Proposed} \\
\midrule
airfoil       &        \cellcolor{red!25}3.6 \tiny{$\pm$0.055} &   \cellcolor{green!25}2.788 \tiny{$\pm$0.054} &                         3.751 \tiny{$\pm$0.143} &   \cellcolor{yellow!25}3.214 \tiny{$\pm$1.029} &     14.48 \tiny{$\pm$0.226} &     11.588 \tiny{$\pm$0.23} &   15.163 \tiny{$\pm$1.694} &                     14.644 \tiny{$\pm$6.464} &   14.449 \tiny{$\pm$0.253} &     11.561 \tiny{$\pm$0.22} &    15.397 \tiny{$\pm$1.303} &                       17.391 \tiny{$\pm$8.0} \\
autos         &                        0.248 \tiny{$\pm$0.007} &                       0.452 \tiny{$\pm$0.026} &                         0.259 \tiny{$\pm$0.008} &    \cellcolor{yellow!25}0.19 \tiny{$\pm$0.041} &     0.305 \tiny{$\pm$0.009} &     0.487 \tiny{$\pm$0.008} &    0.306 \tiny{$\pm$0.008} &    \cellcolor{red!25}0.229 \tiny{$\pm$0.065} &    0.309 \tiny{$\pm$0.018} &      0.49 \tiny{$\pm$0.011} &     0.306 \tiny{$\pm$0.008} &  \cellcolor{green!25}0.172 \tiny{$\pm$0.033} \\
boston        &    \cellcolor{green!25}1.604 \tiny{$\pm$0.185} &                       3.039 \tiny{$\pm$0.165} &       \cellcolor{red!25}1.966 \tiny{$\pm$1.012} &   \cellcolor{yellow!25}1.779 \tiny{$\pm$1.421} &    11.649 \tiny{$\pm$0.114} &    21.064 \tiny{$\pm$0.489} &   11.633 \tiny{$\pm$0.052} &                     13.837 \tiny{$\pm$7.176} &   11.712 \tiny{$\pm$0.085} &    21.036 \tiny{$\pm$0.829} &    11.666 \tiny{$\pm$0.053} &                      9.967 \tiny{$\pm$1.239} \\
breastcancer  &   \cellcolor{yellow!25}0.271 \tiny{$\pm$0.013} &                       0.469 \tiny{$\pm$0.013} &       \cellcolor{red!25}0.275 \tiny{$\pm$0.015} &    \cellcolor{green!25}0.172 \tiny{$\pm$0.059} &      0.345 \tiny{$\pm$0.02} &      0.513 \tiny{$\pm$0.01} &    0.349 \tiny{$\pm$0.025} &                      0.305 \tiny{$\pm$0.097} &    0.348 \tiny{$\pm$0.025} &      0.52 \tiny{$\pm$0.022} &     0.356 \tiny{$\pm$0.027} &                        0.323 \tiny{$\pm$0.1} \\
california    &    \cellcolor{red!25}112.311 \tiny{$\pm$5.921} &                     117.303 \tiny{$\pm$7.352} &  \cellcolor{yellow!25}107.375 \tiny{$\pm$1.904} &  \cellcolor{green!25}91.654 \tiny{$\pm$32.651} &  413.289 \tiny{$\pm$25.963} &  410.402 \tiny{$\pm$21.844} &  375.299 \tiny{$\pm$6.141} &                  440.169 \tiny{$\pm$216.265} &   409.07 \tiny{$\pm$24.33} &  405.269 \tiny{$\pm$20.901} &   370.862 \tiny{$\pm$4.876} &                  799.192 \tiny{$\pm$270.568} \\
concrete      &   \cellcolor{yellow!25}1.627 \tiny{$\pm$0.015} &   \cellcolor{green!25}1.618 \tiny{$\pm$0.032} &                         1.674 \tiny{$\pm$0.048} &      \cellcolor{red!25}1.642 \tiny{$\pm$0.682} &     6.884 \tiny{$\pm$0.029} &     5.309 \tiny{$\pm$0.066} &    6.994 \tiny{$\pm$0.194} &                      5.469 \tiny{$\pm$1.727} &     6.909 \tiny{$\pm$0.06} &     5.313 \tiny{$\pm$0.032} &      7.03 \tiny{$\pm$0.176} &                      5.407 \tiny{$\pm$0.978} \\
concreteslump &      \cellcolor{red!25}0.256 \tiny{$\pm$0.015} &                       0.466 \tiny{$\pm$0.011} &    \cellcolor{yellow!25}0.245 \tiny{$\pm$0.009} &    \cellcolor{green!25}0.166 \tiny{$\pm$0.026} &     0.277 \tiny{$\pm$0.009} &     0.483 \tiny{$\pm$0.012} &    0.279 \tiny{$\pm$0.006} &                      0.262 \tiny{$\pm$0.081} &    0.276 \tiny{$\pm$0.012} &      0.49 \tiny{$\pm$0.026} &     0.279 \tiny{$\pm$0.007} &                      0.317 \tiny{$\pm$0.142} \\
energy        &   \cellcolor{yellow!25}1.037 \tiny{$\pm$0.018} &   \cellcolor{green!25}0.926 \tiny{$\pm$0.024} &       \cellcolor{red!25}1.048 \tiny{$\pm$0.017} &                        2.669 \tiny{$\pm$0.093} &     3.763 \tiny{$\pm$0.039} &     2.946 \tiny{$\pm$0.176} &    3.818 \tiny{$\pm$0.054} &                     10.424 \tiny{$\pm$0.478} &    4.015 \tiny{$\pm$0.662} &     3.054 \tiny{$\pm$0.502} &     3.848 \tiny{$\pm$0.139} &                     10.186 \tiny{$\pm$0.187} \\
gas           &  \cellcolor{yellow!25}10.485 \tiny{$\pm$0.062} &   \cellcolor{green!25}8.494 \tiny{$\pm$0.103} &      \cellcolor{red!25}10.918 \tiny{$\pm$0.897} &                         11.9 \tiny{$\pm$5.833} &     40.42 \tiny{$\pm$0.245} &    35.966 \tiny{$\pm$0.266} &   41.341 \tiny{$\pm$0.632} &                    51.858 \tiny{$\pm$34.278} &   40.394 \tiny{$\pm$0.205} &     35.843 \tiny{$\pm$0.19} &    41.643 \tiny{$\pm$1.002} &                    111.635 \tiny{$\pm$0.403} \\
machine       &      \cellcolor{red!25}0.266 \tiny{$\pm$0.008} &                       0.473 \tiny{$\pm$0.025} &                          0.271 \tiny{$\pm$0.01} &    \cellcolor{green!25}0.225 \tiny{$\pm$0.064} &     0.358 \tiny{$\pm$0.022} &      0.52 \tiny{$\pm$0.008} &     0.376 \tiny{$\pm$0.01} &  \cellcolor{yellow!25}0.248 \tiny{$\pm$0.05} &    0.361 \tiny{$\pm$0.025} &      0.52 \tiny{$\pm$0.008} &      0.371 \tiny{$\pm$0.01} &                      0.295 \tiny{$\pm$0.074} \\
pendulum      &     \cellcolor{green!25}0.801 \tiny{$\pm$0.02} &  \cellcolor{yellow!25}0.803 \tiny{$\pm$0.013} &       \cellcolor{red!25}0.815 \tiny{$\pm$0.026} &                        0.898 \tiny{$\pm$0.362} &      2.59 \tiny{$\pm$0.027} &     2.132 \tiny{$\pm$0.028} &    2.658 \tiny{$\pm$0.044} &                      2.327 \tiny{$\pm$0.895} &    2.597 \tiny{$\pm$0.028} &     2.138 \tiny{$\pm$0.043} &     2.699 \tiny{$\pm$0.088} &                      2.311 \tiny{$\pm$0.522} \\
servo         &    \cellcolor{yellow!25}0.257 \tiny{$\pm$0.01} &                       0.469 \tiny{$\pm$0.023} &       \cellcolor{red!25}0.313 \tiny{$\pm$0.046} &    \cellcolor{green!25}0.255 \tiny{$\pm$0.147} &     0.316 \tiny{$\pm$0.012} &     0.499 \tiny{$\pm$0.013} &    0.415 \tiny{$\pm$0.087} &                      0.368 \tiny{$\pm$0.196} &    0.317 \tiny{$\pm$0.012} &     0.504 \tiny{$\pm$0.008} &     0.397 \tiny{$\pm$0.049} &                      0.344 \tiny{$\pm$0.187} \\
sml           &     \cellcolor{red!25}26.593 \tiny{$\pm$0.114} &   \cellcolor{green!25}22.19 \tiny{$\pm$0.131} &                        30.462 \tiny{$\pm$3.213} &   \cellcolor{yellow!25}24.86 \tiny{$\pm$8.095} &   107.036 \tiny{$\pm$0.437} &    97.599 \tiny{$\pm$0.902} &  120.946 \tiny{$\pm$9.981} &                   241.597 \tiny{$\pm$83.722} &  106.985 \tiny{$\pm$0.788} &    97.702 \tiny{$\pm$0.873} &  121.981 \tiny{$\pm$11.739} &                    292.416 \tiny{$\pm$1.145} \\
stock         &      \cellcolor{red!25}0.655 \tiny{$\pm$0.022} &                       0.731 \tiny{$\pm$0.046} &     \cellcolor{yellow!25}0.648 \tiny{$\pm$0.02} &    \cellcolor{green!25}0.447 \tiny{$\pm$0.131} &     1.812 \tiny{$\pm$0.019} &     2.154 \tiny{$\pm$0.201} &     1.851 \tiny{$\pm$0.03} &                      1.197 \tiny{$\pm$0.304} &    1.808 \tiny{$\pm$0.013} &     1.606 \tiny{$\pm$0.806} &     1.847 \tiny{$\pm$0.032} &                      1.193 \tiny{$\pm$0.354} \\
wine          &      \cellcolor{red!25}3.972 \tiny{$\pm$0.027} &   \cellcolor{green!25}3.092 \tiny{$\pm$0.053} &                          4.267 \tiny{$\pm$0.43} &   \cellcolor{yellow!25}3.528 \tiny{$\pm$1.004} &      16.1 \tiny{$\pm$0.278} &      13.042 \tiny{$\pm$0.1} &   16.865 \tiny{$\pm$0.971} &                     16.682 \tiny{$\pm$5.333} &   16.133 \tiny{$\pm$0.316} &    13.067 \tiny{$\pm$0.083} &    16.875 \tiny{$\pm$1.093} &                    27.033 \tiny{$\pm$11.773} \\
yacht         &   \cellcolor{yellow!25}0.302 \tiny{$\pm$0.008} &                       0.487 \tiny{$\pm$0.008} &       \cellcolor{red!25}0.305 \tiny{$\pm$0.008} &      \cellcolor{green!25}0.3 \tiny{$\pm$0.203} &     0.767 \tiny{$\pm$0.022} &     0.807 \tiny{$\pm$0.027} &    0.787 \tiny{$\pm$0.023} &                      0.791 \tiny{$\pm$0.272} &     0.76 \tiny{$\pm$0.021} &      0.783 \tiny{$\pm$0.01} &     0.792 \tiny{$\pm$0.039} &                      1.981 \tiny{$\pm$0.039} \\
\bottomrule
\end{tabular}
\end{table}

## Attribtion

1. Gradient Cobra is taken from [Gradient Cobra](https://github.com/hassothea/gradientcobra)
2. For base models [SciKit Learn](https://scikit-learn.org/stable/index.html) is used.