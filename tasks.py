import json
import numpy as np
seed_generator = 1
#from multiprocessing import Pool, freeze_support
#from optimparallel import minimize_parallel
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import ConstantKernel, Matern
noise = 0.8
# Gaussian process with Mat??rn kernel as surrogate model
#m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
                noise_level=1, noise_level_bounds=(1e-5, 1e1)
            )
gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)
from celery import Celery

#from celery import current_app
#from celery.contrib.methods import task_method
#CELERY_ACCEPT_CONTENT = ['pickle']
REDIS_BASE_URL = 'redis://localhost:6379'

app = Celery('tasks', broker='redis://localhost:6379/0')
#### This is currently working
#app = Celery('tasks', broker='pyamqp://guest@localhost//')


#app = Celery('tasks', backend='rpc://', broker='pyamqp://')
#app = Celery('tasks', backend='redis://localhost', broker='pyamqp://')
#####
## Author: PM 23/06/2021
## Generic Biobjective optimiser but can be easily adapted to multiobjective by 
## declaring and adding objective functions and different parameters 
#####

import math
import numpy as np
import pandas as pd
import pickle
from scipy.optimize import minimize
import sys
import pdb
import matplotlib
import matplotlib.pyplot as plt
import plotly.figure_factory as ff  # type: ignore
seed_generator = 1
#import tensorflow as tf
#import autograd.numpy as np
#from autograd import grad, jacobian, hessian

#tf.enable_eager_execution()

values = []
ValuesOfParam = []
#param=[]
method = 'SLSQP'
bounds = [[0.5326,2.2532]] # [[-1.e2,1.e2]] # 
gamma_step = 1/(3*(np.sqrt(200)))
font = {'family' : 'normal',
'weight' : 'bold',
'size'   : 22}
objectives = 2
a = 1  # constant that does not influence optimised result (1 in article)
f = [0, 0]  # normalised objective function value
w = [0, 0]  # normalised weight function
n =(objectives)+3
beta = 1.1
restrictivePref  =False
init_guess = np.random.uniform( bounds[0][0],bounds[0][1] ) #1
storeP = []#for helping me out.
P = None
args = None



import xlwt

wb = xlwt.Workbook()
worksheet = wb.add_sheet('Sheet 1')

#LIDAR_RES = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def expected_improvement( X, X_sample, Y_sample, gpr, xi=0.3):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [1]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei.flatten()
####
## Use this when you use optimparallel
####
def min_obj( X,acquisition, X_sample,Y_sample,gpr,dim):

    #Minimization objective is the negative acquisition function
    return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
def propose_location( acquisition,args, X_sample, Y_sample, gpr, bounds, n_restarts=1):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val =0
    min_x = None
    
    #####
    ## Use this in normal scipy minimize
    #####
    def min_obj(X):
        #Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    #pdb.set_trace()
    for x0 in np.random.uniform(bounds[0][0], bounds[0][1], size=(n_restarts, dim)):
        #res = minimize_parallel (self.min_obj, x0=1, args = (acquisition,X_sample,Y_sample,gpr,dim) ,bounds=self.bounds ,options = {'maxiter': 1000,'disp':False } )  
        res = minimize(min_obj, x0=init_guess,  bounds=bounds, method='L-BFGS-B',options = {'maxiter': 500,'disp':False })
        #res = minimize( self.normalisedWeightFunction,np.random.uniform( self.bounds[0][0],self.bounds[0][1] ),method=self.method\
        #    ,bounds=self.bounds, args= (args,),  options = {'maxiter': 1000,'disp':False})     
        if res['fun'] < min_val:
            min_val = res['fun'][0]
            min_x = res['x']     

    return res['x'].reshape(-1, 1)

          



@app.task()
def runAggregateObjFunc(serialisable):
    global ValuesOfParam
    args = json.loads(serialisable)
    gamma1 = args['gamma1']
    gamma_step = args['gamma_step']
    bounds = args['bounds']
    min_obj =  (  args['min_obj'] )
    max_obj =  ( args['max_obj'] )
    #portfolios = args['portfolios']
    #port_shocks = args['port_shocks']
    S_f = args['S_f']
    S_p = args['S_p']
    E = args['E']
    P_0 = args['P_0']
    
    #individual_returns = args['individual_returns']
    frontier = []
    #pdb.set_trace()
    for gamma2 in np.arange(0, 1 - gamma1 + gamma_step, gamma_step):
        #gamma3 = 1-gamma1-gamma2
        
        # Create gamma array
        gamma = np.array([[gamma1, 0 ], [0, gamma2 ] ])
        gamma_cmp = np.array([[1-gamma1, 0 ], [0, 1-gamma2 ] ])
        #print (gamma.shape)
        #print (gamma_cmp.shape)
        #print (min_obj[0])

        min_obj = np.array(min_obj[0]).reshape ( (1,2))
        max_obj = np.array(max_obj[0]).reshape ( (1,2))

        # Create Translation Vector
        S_t = np.dot(min_obj, gamma) + np.dot(max_obj,gamma_cmp)
        S_tf = S_f + S_t.T + S_p
        #print (P_0.shape , E.shape,S_f.shape,S_t.shape,S_p.shape,S_tf.shape)
        P = np.dot(S_tf, E) + P_0
        args['P'] = P
        global seed_generator
        want_random_init = True
        np.random.seed(seed_generator)
        seed_generator += 1
        n_iter = 2
        # Initialize samples
        X_sample = np.array([0.6]).reshape(1,1)
        Y_sample = np.array([math.sin( 0.6 )] ).reshape(1,1)

        #if want_random_init:
        #    init_guess = np.array(np.random.dirichlet(np.ones( len(portfolios)), size=1))
        #init_guess =  np.array( [0.053,0.154,0.525,0.051,0.052,0.163])
        #var1 = tf.Variable(0.0)
        # Create an optimizer with the desired parameters.
        #opt = tf.keras.optimizers.SGD(learning_rate=0.1)
        #for i in range(100):
        #    opt_op = opt.minimize(self.normalisedWeightFunction , var_list=[var1])
        #start = tf.constant([0.9])  # Starting point for the search.
        #optim_results = tfp.optimizer.bfgs_minimize(
        #    quadratic_loss_and_gradient, initial_position=start, tolerance=1e-8)

        #pdb.set_trace()
        GaAlgo  =0

        #jacobian_  = jacobian(self.normalisedWeightFunction)
        if not GaAlgo:
            try:
                for i in range(n_iter):
                    # Update Gaussian process with existing samples
                    #pdb.set_trace()
                    gpr.fit(X_sample, Y_sample)
                    
                    # Obtain next sampling point from the acquisition function (expected_improvement)
                    X_next = propose_location(expected_improvement,args, X_sample, Y_sample, gpr, bounds)
                    #print (i)
                    # Obtain next noisy sample from the objective function
                    Y_next = normalisedWeightFunction (X_next, P)
                    #print (X_next)

                    #param.append(X_next) 
                    # Add sample to previous samples
                    X_sample = np.vstack((X_sample, X_next))
                    Y_sample = np.vstack((Y_sample, Y_next))
                    f = open("output.txt", "a+")
                    #for d in data:
                    f.write(f"{ float(X_next)  }\n")

                    f.close()

            except ValueError as e:
                result = {"success": False, "message": str(e)}
        else:
            try:
                pass
                #result = differential_evolution( self.normalisedWeightFunction, self.bounds, seed = 1, maxiter=1000, \
                #    disp=False, args=args)
            except ValueError:
                result = {"success": False}

        #ValuesOfParam.append(result['x'])  
    #print(ValuesOfParam)        
    return X_next                  


def normalisedWeightFunction(weights ,args ):

    weights = np.array(weights)
    i = 0

    o1 = get_ret_heuristic(weights) [1]    
    o2 =  get_ret_heuristic(weights) [2]    
    order = [o1, o2]
    num = 0
    denom = 0
    #pdb.set_trace()
    P=args
    #pdb.set_trace()
    storeP.append(P)
    for i in range(len(P)):

        val = order[i]
        pref = P[i]
            
        upper = 0
        lower = 0
        #print (pref)
        #pdb.set_trace()
        lamb= np.random.uniform(0,1,1)
        if(val > pref[4]):
            return sys.maxsize
            # print("specific result: "+ str(abs(val - pref[4]) * 1000))
        elif(val < pref[0]):
            #f[i] = a* np.random.uniform(0,np.exp(  (val-pref[0])/(pref[1]-pref[0])) ,1)  
            f[i] = a* np.exp(  (val-pref[0])/(pref[1]-pref[0]))
        else:
            for j in range(len(pref)):
                preference = pref[j]
                if(preference > val):
                    break
            k = j + 1
            upper = pref[j]
            lower = pref[j-1]
            f[i] = (k-1)*a + a*(val-lower)/(upper-lower)

        # Calculate weight
        if(f[i] != 200):
            w[i] = (beta*(n-1))**(f[i]/2)
        else:
            w[i] = sys.maxsize

    for i in range(len(w)):
        num += w[i]*f[i]
        denom += w[i]
    outVal = num/denom
    values.append( outVal)  
    
    return outVal

  
def get_ret_heuristic( weights):
    weights = np.array(weights)
    obj1 = np.sin( weights )
    obj2 = 1 - np.sin(weights)**7
    #obj1 = weights**2
    #obj2 = (weights-2)**2
    return np.array([ 0 , obj1, obj2,0])
def minimiseObj2( weights):
    return  get_ret_heuristic(weights)[2]
def minimiseObj1(  weights):
    return  get_ret_heuristic(weights)[1]

def maximiseObj1( weights):
    return - get_ret_heuristic(weights)[1]

def maximiseObj2( weights):
    return -get_ret_heuristic(weights)[2]
def constraint1(x):
    return 1.2532-x
def constraint2( x):
    return x-0.5326
def getMinMaxCaseStudy():
    #where the inequalities are of the form C_j(x) >= 0.

    con1 = ({'type': 'ineq', 'fun':  constraint1},
            {'type': 'ineq', 'fun': constraint2 })
    columns = ['obj1', 'obj2']
    index = ['min', 'max']
    minMax = pd.DataFrame(index=index, columns=columns, dtype=object)

    minObj2 = minimize( minimiseObj2, init_guess, method=method,
                    bounds= bounds )['x']
    results = get_ret_heuristic(minObj2) [2]  
    
    minObj2 = results 

    maxObj2 = minimize(maximiseObj2, init_guess, method=method,
                    bounds=bounds )['x']
    results =  get_ret_heuristic(maxObj2) [2]  
    maxObj2 = results 

    minObj1 = minimize(minimiseObj1, init_guess,
                        method=method, bounds=bounds )['x']
    minObj1 = get_ret_heuristic(minObj1) [1]  

    maxObj1 = minimize(maximiseObj1, init_guess,
                        method=method, bounds=bounds )['x']
    maxObj1 = get_ret_heuristic(maxObj1) [1]   
    
    if restrictivePref:

        minMax['obj2']['min'] = 0.4 #minObj2
        minMax['obj2']['max'] = 0.8 #maxObj2
        minMax['obj1']['min'] =  0.5 # minObj1
        minMax['obj1']['max'] = 0.7 #maxObj1
    else: 
        minMax['obj2']['min'] = minObj2
        minMax['obj2']['max'] =  maxObj2
        minMax['obj1']['min'] =   minObj1
        minMax['obj1']['max'] = maxObj1
    
    return (minMax)

