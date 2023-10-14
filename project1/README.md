# fys-stk4155
collaboration repos for fys-stk4155

Command line arguments to reproduce the plots of Results.
Section Ordinary least squares:

Figure 1:
python3 project1.py 10 300 0.1 1

Figure 2:
python3 project1.py 10 300 0.1 10

Figure 3:
python3 project1.py 10 30 0.1 100
python3 project1.py 10 300 0.1 100
python3 project1.py 10 3000 0.1 100

No figure - investigation of sigma
python3 project1.py 10 30 0.0 100
python3 project1.py 10 300 0.0 100
python3 project1.py 10 300 0.1 100
python3 project1.py 10 300 0.2 100

Part b) Ridge on Franke function
python3 ridge_franke_analysis.py 15 500 0.1 200 0.000000001
python3 ridge_franke_analysis.py 15 500 0.1 200 0.0000001
python3 ridge_franke_analysis.py 15 500 0.1 200 0.00001
python3 ridge_franke_analysis.py 15 500 0.1 200 0.001
python3 ridge_franke_analysis.py 15 500 0.1 200 0.1

Part c) Lasso on Franke function
python3 lasso_franke_analysis.py 15 500 0.1 10 0.1
python3 lasso_franke_analysis.py 15 500 0.1 10 0.01
python3 lasso_franke_analysis.py 15 500 0.1 10 0.001
python3 lasso_franke_analysis.py 15 500 0.1 10 0.0001
python3 lasso_franke_analysis.py 15 500 0.1 10 0.00001
