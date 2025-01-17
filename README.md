
# Quantum Threads

An end to end under water animal identification machine learning model using Quantum kernels along with Classical pretrained models.
This project comprises a brief research on three popular classical models ResNet18 , ResNet50 ,InceptionV3 and its quantum counter parts Hybrid Quantum classical ResNet18 | ResNet50 | InceptionV3.

# Base solution

* To provide a web app backboned by Hybrid quantum classical CNN models 
* Hybrid Quantum ML model will use  PQC as Quantum kernels along with pretrained classical models
* Whole model will be trained on UAD-2023 dataset containing >13k images for 23 classes
* Entire model is made learnable , quantum kernels will use classical gradient descent for optimization

# What's Innovative ?

## Distributed kernel processing
* Distributing all kernels across multiple cloud QPU's for parallelism

## Adjoint differentiation
* Using reversible nature of quantum circuits to compute gradients , thus bringing time complexity from exponential to constant







## Demo
![resnet-50](https://github.com/nagarajRPoojari/Quantum-Threads/assets/116948655/0108ce20-d8f5-434f-8ee7-1b589b022b9c)
![qpu-hardwere](https://github.com/nagarajRPoojari/Quantum-Threads/assets/116948655/c00cbb9e-2e12-41e0-a017-b1b2d0a43a2c)



## Environment Variables

If you want to run inference on real IBM Quantum device you need to add IBM Quantum token to config.toml file

`API_KEY`




## Run locally

Clone this repo

```bash
  git clone https://github.com/nagarajRPoojari/Quantum-Threads.git
```
install requirements
```bash
    pip install -r requirements.txt
```
start streamlit server
```
    streamlit run main.py
```
    
## Authors

- [@nagarajRPoojari](https://github.com/nagarajRPoojari)
- [@nithinmkannal](https://github.com/nithinmkannal)
- [@Varun-rocky](https://github.com/Varun-rocky)



## Tech Stack

* Tensorflow
* Pennylane
* Qiskit
* Amazon braket
* streamlit 
* AWS


## License

[MIT](https://choosealicense.com/licenses/mit/)


