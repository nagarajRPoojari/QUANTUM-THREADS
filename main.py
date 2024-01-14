import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
from QuantumThreads.config.configuration import ConfigurationManager
from QuantumThreads.pipeline import Pipeline , merge
import time
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Quantum",page_icon=":rocket:",layout="wide")
st.title("Quantum Threads")

@st.cache_resource()
def load_model():
    config=ConfigurationManager().get_model_trainer_config()
    pipeline=Pipeline(config=config)
    return pipeline


pipeline=load_model()


c1,c2=st.columns(2)

selected_model = c2.selectbox("Select model", ["Resnet50", "Hybrid-quantum-Resnet50"])


if selected_model == "Resnet50":
    device = ["CPU"]
elif selected_model == "Hybrid-quantum-Resnet50":
    device = ["IBM simulator-qiskit.aer", "pennylane simulator-default.qubit","IBM Qiskit kyoto(Hardware)"]
else:
    device = []
    

selected_model = c2.selectbox("Select device", device)


image=c1.file_uploader("Upload image",type=["jpg",'png','jpeg'])
if image !=None and selected_model=="CPU":
    with c2.status("Loading classical model...", expanded=True) as status:
        res,prob=pipeline.inference(image)
        status.update(label="Inference completed!", state="complete", expanded=False)

    c1.write(res)
    c1.image(image, caption="Uploaded Image",width=400)

    c2.bar_chart(prob,x='class')
    
elif image!=None and selected_model!="IBM Qiskit kyoto(Hardware)":
    with c2.status("Loading statevector simulator...", expanded=True) as status:
        res,prob=pipeline.inference(image,device='quantum')
        status.update(label="Inference completed!", state="complete", expanded=False)
        
    c1.write(res)
    c1.image(image, caption="Uploaded Image",width=400)   
    c2.bar_chart(prob,x='class')
    
elif image!=None and selected_model=="IBM Qiskit kyoto(Hardware)":
    with c2.status("Job added to queue...", expanded=True) as status:
        
        #pipeline for QPU
        dev,circuit=pipeline.inference(image,device='QPU')
        
        
        c2.write(f"{dev.capabilities()}")
        sample_inputs=np.random.rand(1,16)
        sample_weights=np.random.rand(1,16)
        circuit(sample_inputs,sample_weights)
        status.update(label="Inference completed!", state="complete", expanded=False)
        
        

    




















# Training results



st.subheader("Training Results")
_,col,_=st.columns([1,4,1])

col.image('./media/model_image.gif', caption='Hybrid Quantum ResNet50 with parallel processing and adjoint differentiation', output_format='auto')



ops=["classical Resnet50 vs Hybrid Resnet50 ","classical Resnet18 vs Hybrid Resnet18 ","classical inception vs Hybrid inception "]
graph=st.selectbox("select training graph",options=ops)


col1,col2=st.columns(2)


resnet50 = "results/resnet50.csv"
resnet50 = pd.read_csv(resnet50)


hybrid_resnet50 = "results/Hybridresnet50.csv"
hybrid_resnet50 = pd.read_csv(hybrid_resnet50)


resnet18 = "results/resnet18.csv"
resnet18 = pd.read_csv(resnet18)


hybrid_resnet18 = "results/HybridResnet18.csv"
hybrid_resnet18 = pd.read_csv(hybrid_resnet18)


inceptionv3 = "results/inceptionV3.csv"
inceptionv3 = pd.read_csv(inceptionv3)


hybrid_inceptionv3 = "results/HybridInceptionV3.csv"
hybrid_inceptionv3 = pd.read_csv(hybrid_inceptionv3)



    
    

col1.write("Training")
col2.write("validation")
if graph==ops[0]:
    a,b=merge(resnet50,hybrid_resnet50,'resnet50')
    
    col1.line_chart(a)
    col2.line_chart(b)
elif graph==ops[1]:
    a,b=merge(resnet18,hybrid_resnet18,'resnet18')
    
    col1.line_chart(a)
    col2.line_chart(b)

elif graph==ops[2]:
    a,b=merge(inceptionv3,hybrid_inceptionv3,'InceptionV3')
    
    col1.line_chart(a)
    col2.line_chart(b)
  


col1.write("Parallel vs Sequential")
parallevswithout=pd.read_csv('./results/parallelVSwithoutParallel.csv')
col1.line_chart(parallevswithout)
col2.write("Adjoint vs Parameter Shift")
adjointvsparametershift=pd.read_csv('./results/adjointVSparametershift.csv')
col2.line_chart(adjointvsparametershift)
    

