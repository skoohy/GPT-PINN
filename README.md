# Generative Pre-Trained Physics-Informed Neural Network Implementation

## GPT-PINN: Generative Pre-Trained Physics-Informed Neural Networks toward non-intrusive Meta-learning of parametric PDEs

### Yanlai Chen<sup>1</sup>, Shawn Koohy<sup>1</sup>

### Update (6/5/2024): The new Klein-Gordon code has been updated, and the original code is still available in the Klein-Gordon folder 
### Update (6/5/2024): The new Burgers' code has been updated, and the original code is still available in the Burgers folder 
### Update (5/20/2024): The code is currently being further optimized for speed and readability  

## Paper Links: [arXiv](https://arxiv.org/abs/2303.14878) | [ResearchGate](https://www.researchgate.net/publication/369556903_GPT-PINN_Generative_Pre-Trained_Physics-Informed_Neural_Networks_toward_non-intrusive_Meta-learning_of_parametric_PDEs) | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168874X23001403)

### See also TGPT-PINN: Nonlinear model reduction with transformed GPT-PINNs ([GitHub](https://github.com/DuktigYajie/TGPT-PINN))

![Image 1](fig/GPT-PINN.png)
*GPT-PINN Architecture*

## Talk and Presentations
[DDPS Seminar at the Lawrence Livermore Lab](https://www.youtube.com/embed/ODA9Po4FVWA?si=z2p9SkprfGZy4aeZ)

[Brown CRUNCH Group Seminar](https://www.youtube.com/embed/wzHyOHV0ZeE?si=ehWaULam9PYJyFgB)

[Numerical Analysis of Galerkin ROMs Seminar](https://www.youtube.com/embed/KWaWH7xeVEg?si=OqtmATD2fmvMSRV1)

## Abstract: 
<em>Physics-Informed Neural Network (PINN) has proven itself a powerful tool to obtain the numerical solutions of nonlinear partial differential equations (PDEs) leveraging the expressivity of deep neural networks and the computing power of modern heterogeneous hardware. However, its training is still time-consuming, especially in the multi-query and real-time simulation settings, and its parameterization often overly excessive. In this paper, we propose the Generative Pre-Trained PINN (GPT-PINN) to mitigate both challenges in the setting of parametric PDEs. GPT-PINN represents a brand-new meta-learning paradigm for parametric systems. As a network of networks, its outer-/meta-network is hyper-reduced with only one hidden layer having significantly reduced number of neurons. Moreover, its activation function at each hidden neuron is a (full) PINN pre-trained at a judiciously selected system configuration. The meta-network adaptively “learns” the parametric dependence of the system and “grows” this hidden layer one neuron at a time. In the end, by encompassing a very small number of networks trained at this set of adaptively-selected parameter values, the meta-network is capable of generating surrogate solutions for the parametric system across the entire parameter domain accurately and efficiently.</em>

</sub></sub><sub>1</sup> University of Massachusetts Dartmouth, Department of Mathematics, North Dartmouth, MA</sub></sub><br>

## Requirements:
```
Python     = 3.9.12
NumPy      = 1.23.4
PyTorch    = 1.11.0
TensorFlow = 2.10.0
Matplotlib = 3.6.2
```
Combinations of different package versions (recent ones) will most likely be able to run the code with little to no change.  

## GPU and CPU Support:
The code was implemented with the intention of computation to be primarily performed on the GPU. CPU computation can be done however, it will take much longer. 

## Usage:
The Klein-Gordon, Allen-Cahn, and Burgers' equation files are currently available. Running `KG_main.py`, `B_main.py`, or `AC_main.py` (with the other files in the folder located in the respective directory) will begin the training of the full-PINN and GPT-PINN, growing the GPT-PINN hidden layer size from 1 to 15 (Klein-Gordon) or 9 (Burgers' and Allen-Cahn). The Final GPT-PINN is then tested on various parameters and the results of training and testing can visualized using the plotting files (`KG_plotting.py`, `B_plotting.py`, or `AC_plotting.py`). Various parameters within the PINN or GPT-PINN can easily be changed in the main files. As a default setting, once the total number of neurons is achieved, the GPT-PINN is trained once more in order to find the largest loss obtained using the final number of neurons. This is done to give more information  about the final state of the GPT-PINN.

![Image 2](fig/KG_t1.png)

*Klein-Gordon Run Times*

![Image 2](fig/B_t1.png)

*Burgers' Run Times*

![Image 2](fig/AC_t1.png)

*Allen-Cahn Run Times*

## Citation:
Below you can find the Bibtex citation:
```
@article{chen2024gpt,
  title={GPT-PINN: Generative Pre-Trained Physics-Informed Neural Networks toward non-intrusive Meta-learning of parametric PDEs},
  author={Chen, Yanlai and Koohy, Shawn},
  journal={Finite Elements in Analysis and Design},
  volume={228},
  pages={104047},
  year={2024},
  publisher={Elsevier}
}
```
