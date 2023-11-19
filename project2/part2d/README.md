Reproducing results of report:

For the final/main results presented last under Neural network classification
python3 neural_net_classification.py
python3 neural_net_classification_tuning.py

For the results right before that some edit of code is needed (we did not spend time to parameterize everything). 
In neural_net_classification.py 
change
network.add_layer(32)
to
network.add_layer(12)
in two places.

In neural_net_classification_tuning.py
change
eta_vals = [ 0.0001, 0.0005, 0.001, 0.005, 0.01]
to
eta_vals = [ 0.00001, 0.00005, 0.0001, 0.0005, 0.001]

and rerun.

And if you want to try Sigmoid and ReLU - the results we did not even show in report, but mentioned we had tested. Then you need to edit these files and replace where it says
'leakyrelu'
with 'relu' or 'sigmoid'
