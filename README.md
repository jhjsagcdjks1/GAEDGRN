
<font size="5"><strong>Reconstruction of gene regulatory networks based on gravity-inspired graph autoencoders</strong></font>
![image](https://github.com/jhjsagcdjks1/GAEDGRN/blob/master/GAEDGRN/Framework.png)
***
**Dependencies**
• 'networkx==2.2'；  
• 'numpy==1.16.0'；  
• 'scikit-learn==0.19.2'；  
• 'scipy==1.0.0'；  
• 'tensorflow==1.4'
***
**Usage**
Preparing for gene expression profiles and gene-gene adjacent matrix

GAEDGRN integrates gene expression matrix (N×M) with prior gene topology (N×N) to learn low-dimensional vertorized representations with supervision.Please convert the gene expression matrix and prior gene topology into the ".npz" format using the data preprocessing code (as shown in the demo), and then run them in the GAEDGRN model.

Command to run GAEDGRN
python train.py
