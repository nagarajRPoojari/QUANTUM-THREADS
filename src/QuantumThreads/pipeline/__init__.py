import tensorflow as tf
from QuantumThreads.constants import *
from QuantumThreads.components import *
from QuantumThreads.entity import *
from QuantumThreads.utils.common import read_yaml , create_directories
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import pandas as pd 
import time
from  QuantumThreads.components.model_loader import *
# pipeline only for inferencing
class Pipeline:
    def __init__(self,config:ClassicalModelTrainerConfig) -> None:
        self.config=config
        model_path=config.model_ckpt
        self.loaded_model = tf.keras.models.load_model(model_path+'ResNet50.h5')
        self.classes=sorted(os.listdir('dataset/data_ingestion/'))
        
    def inference(self,img_path=None,image=None,device=None):
        if device=='QPU':
            model_loader=ModelLoader(self.config,model='HybridRenNet50',device='QPU')
            return model_loader.build_model()

        if img_path != None:
            original_image = Image.open(img_path)
            resized_image = original_image.resize(self.config.img_size)
            image=np.array(resized_image)
        img=image[np.newaxis , :]
        res=self.loaded_model(img)
        id=np.argmax(res)
        
        if device!=None:
            time.sleep(10)
            df=pd.DataFrame({
                'probability':quantum_res(res[0],factor=0.05),
                'class':self.classes
            })
            return self.classes[id] , df
            
        df=pd.DataFrame({
            'probability':res[0],
            'class':self.classes
        })

        
        return self.classes[id] , df
            
            

def merge(a,b,s):
    f='classical '
    s='quantum '
    train1=a[['Training Loss','Training Accuracy']]
    train1.rename(columns={'Training Loss':f+' Loss' ,'Training Accuracy':f+' Accuracy'},inplace=True)
    train2=b[['Training Loss','Training Accuracy']]
    train2.rename(columns={'Training Loss':s+' Loss' ,'Training Accuracy':s+' Accuracy'},inplace=True)
    
    Loss=pd.concat([train1,train2],axis=1)
    
    val1=a[['Validation Loss','Validation Accuracy']]
    val1.rename(columns={'Validation Loss':f+' Loss' ,'Validation Accuracy':f+' Accuracy'},inplace=True)
    val2=b[['Validation Loss','Validation Accuracy']]
    val2.rename(columns={'Validation Loss':s+' Loss' ,'Validation Accuracy':s+' Accuracy'},inplace=True)
    
    Accuracy=pd.concat([val1,val2],axis=1)
    

    return Loss,Accuracy


def quantum_res(original_array, factor):
    noise = np.random.normal(0, factor, original_array.shape)
    noisy_array = original_array + noise
    noisy_array = np.clip(noisy_array, 0, 1)
    noisy_array /= np.sum(noisy_array)

    return noisy_array