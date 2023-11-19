Reproducing results of report:

Run command:
python3 testing_neural_net.py

To reproduce all results in report, some source code modification is needed (we did not spend time to parameterize everything):

A - to run with only one hidden layer:
Change rows
network.add_layer(12)
network.add_layer(12)
to
network.add_layer(60)

B - to try the not adviced design matrix with x^2 terms:
Change rows
X = create_design_matrix(x, 1)
to
X = create_design_matrix(x, 2)

C - try only 10 data points, 20 nodes
Change 
n = 400
to
n = 10
and rows
network.add_layer(12)
network.add_layer(12)
to
network.add_layer(20)

D - learning so large it diverges:
change row 
learning_rate = 0.0001 
to
learning_rate = 0.01 


