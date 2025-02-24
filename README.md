
<font size="5"><strong>Reconstruction of gene regulatory networks based on gravity-inspired graph autoencoders</strong></font>
![image](https://github.com/jhjsagcdjks1/GAEDGRN/blob/master/GAEDGRN/Framework.png)
***
**Dependencies**
• 'networkx==2.2'；  
• 'numpy'；  
• 'scikit-learn'；  
• 'scipy'；  
• 'tensorflow==1*'
***
**Usage**
Preparing for gene expression profiles and gene-gene adjacent matrix

GAEDGRN integrates gene expression matrix (N×M) with prior gene topology (N×N) to learn low-dimensional vertorized representations with supervision.Convert the gene expression matrix and prior gene topology into the `.npz` format as shown in the demo, and then run them in the GAEDGRN model.

Command to run GAEDGRN
 python main.py
