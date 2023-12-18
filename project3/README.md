# fys-stk4155
collaboration repos for fys-stk4155, project 3.

Structure:
We have structured the source code, so that files are in subdirectories part3b for Forward Euler
and part3cd for Neural Network. 
This corresponding to the subtasks of project description at
https://github.com/CompPhysics/MachineLearning/blob/master/doc/Projects/2023/Project2/pdf/Project3.pdf
The report has no references to these as subtasks, this is only in the code structure. 

Reproduce the results of the report:
For subtasks b) 
cd part3b
python3 DiffusionEquation1D.py

For subtask c) and d)
cd part3cd
python3 experiment_nn_diffusion.py

However you'll need to modify the parameters inside experiment_nn_diffusion.py to produce all figures and numbers in report.
The currently hard coded values are only for the optimial setting mentioned at the end with ReLU, 2 layers with 400 nodes.
No parameterization this time - sorry about that!
At least all parameters you need to change are in the very top of the Python file.


