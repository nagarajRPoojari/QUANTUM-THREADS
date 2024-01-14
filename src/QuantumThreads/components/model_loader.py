from QuantumThreads.components.data_transformation import DataTransformation
from QuantumThreads.entity import DataIngetionConfig
from QuantumThreads.entity import *
import tensorflow as tf
from tensorflow.keras.applications import ResNet50 , InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
import pennylane as qml 
import numpy as np 
from qiskit import IBMQ

class ModelLoader:
    def __init__(self,
                 config:ClassicalModelTrainerConfig , model ,device='CPU'):
        self.config=config
        self.model_path=config.model_ckpt
        self.device=device
        
        
    def build_model(self):
        self.loaded_model = tf.keras.models.load_model(self.model_path+'ResNet50.h5')
        print(qml.default_config)
        token='9296d1587687ddf1e944cdff40667ef61563f94d507cb4f1b78623ab038fccbe82132e9bbce8821944c91a572ca525109fefb56221b2d48b33c33fc2fcaf8ad3'
        IBMQ.save_account(token=token, overwrite=True)

        IBMQ.load_account()
        provider = IBMQ.get_provider()
        available_backends = provider.backends(filters=lambda x: x.configuration().n_qubits >= 2 and not x.configuration().simulator)
        sorted_backends = sorted(available_backends, key=lambda x: x.status().pending_jobs)
        least_busy_backend = sorted_backends[0]
        no_qubits=16
        dev=qml.device('qiskit.ibmq', wires=no_qubits , backend=least_busy_backend,ibmqx_token=token)
        @qml.qnode(dev)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(no_qubits))
            qml.BasicEntanglerLayers(weights , wires=range(no_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(no_qubits)]
        return dev , circuit
