from audioop import minmax
from pickletools import float8
from tasks import *
import tasks
import matplotlib
import matplotlib.pyplot as plt
import plotly.figure_factory as ff  # type: ignore
import pickle
import numpy as np

import json
import pandas as pd
from celery.result import AsyncResult
from tasks import ValuesOfParam
from tasks import app
import pdb
from scipy.stats import norm
from scipy.optimize import minimize
class runCelery():

    def __init__(self) -> None:
        pass
        self.gamma_step = 1/(3*(np.sqrt(200)))
        self.bounds = [[0.5326,2.2532]] # [[-1.e2,1.e2]] # 
        self.P = None

    def plot(self,data,minMax ):
        #matplotlib.rc('font', **self.font)
        colorscale = ['#7A4579', '#D56073',
                        'rgb(255, 237, 222)', (1, 1, 0.2), (0.8, 0.8, 0.98)]
        fig, ax =plt.subplots(1,1, figsize=(13,13),sharey=True)

        #ax.scatter(  np.sin(data), 1 - np.sin( data )**7 ,color = 'red')
        #ax.plot(  [minMax['obj1'][0],minMax['obj1'][1] ], [minMax['obj2'][1],minMax['obj2'][0] ],color = 'red')

        ax.set_xlabel( r'$f_1 =\sin \theta$' )
        ax.set_ylabel( r'$f_2 = 1 - \sin^7 \theta$' )
        ax.legend(['pareto solutions'],loc="upper right")
        #theta = np.linspace( 0, 2*math.pi,400)
        theta = np.linspace(-1.e3,1.e3,40000)
        fig, axGeneric =plt.subplots(1,1, figsize=(13,13),sharey=True)
        axGeneric.plot( theta,self.get_ret_heuristic(theta)[1],color = 'blue')
        axGeneric.plot( theta,self.get_ret_heuristic(theta)[2] ,color = 'cyan')
        axGeneric.set_xlim(self.bounds[0])
        axGeneric.set_ylim([0,100])
        #print (funArray)
        #print (fun.getValues())
        #pdb.set_trace()
        #sliceLen = 1000 #len(self.storeP)
        #slicedArr = self.storeP#[:sliceLen]
        #ax.plot( slicedArr[0],slicedArr[1] ,color = 'blue')
        #for i in range( sliceLen ):
        #    temp = slicedArr[i]
        #    ax.plot( temp[0] ,temp[1] )
        dat =  np.array(self.param) 
        #print (max(dat))
        ax.scatter( np.arange(len(dat)) , dat  )
        #print (min(dat))
        axGeneric.scatter(  self.get_ret_heuristic(dat)[1], self.get_ret_heuristic(dat)[2]  ,color = 'magenta')
        axGeneric.plot(  self.get_ret_heuristic(theta)[1] , self.get_ret_heuristic(theta)[2]  ,color = 'red')
        #pdb.set_trace()
        plt.show()
        
def plot(param,minMax=None):
    #matplotlib.rc('font', **font)
    colorscale = ['#7A4579', '#D56073',
                    'rgb(255, 237, 222)', (1, 1, 0.2), (0.8, 0.8, 0.98)]
    fig, ax =plt.subplots(1,1, figsize=(13,13),sharey=True)

    #ax.scatter(  np.sin(data), 1 - np.sin( data )**7 ,color = 'red')
    #ax.plot(  [minMax['obj1'][0],minMax['obj1'][1] ], [minMax['obj2'][1],minMax['obj2'][0] ],color = 'red')

    ax.set_xlabel( r'$f_1 =\sin \theta$' )
    ax.set_ylabel( r'$f_2 = 1 - \sin^7 \theta$' )
    ax.legend(['pareto solutions'],loc="upper right")
    theta = np.linspace( 0, 2*math.pi,400)
    #theta = np.linspace(-1.e3,1.e3,40000)
    fig, axGeneric =plt.subplots(1,1, figsize=(13,13),sharey=True)
    axGeneric.plot( theta,get_ret_heuristic(theta)[1],color = 'blue')
    axGeneric.plot( theta,get_ret_heuristic(theta)[2] ,color = 'cyan')
    #axGeneric.set_xlim(bounds[0])
    #axGeneric.set_ylim([0,100])
    #print (funArray)
    #print (fun.getValues())
    #pdb.set_trace()
    #sliceLen = 1000 #len(storeP)
    #slicedArr = storeP#[:sliceLen]
    #ax.plot( slicedArr[0],slicedArr[1] ,color = 'blue')
    #for i in range( sliceLen ):
    #    temp = slicedArr[i]
    #    ax.plot( temp[0] ,temp[1] )
    dat =  np.array(param) 
    #print (max(dat))
    #pdb.set_trace()

    ax.scatter( np.arange(len(dat)) , dat  )
    #print (min(dat))
    axGeneric.scatter(  get_ret_heuristic(dat)[1], get_ret_heuristic(dat)[2]  ,color = 'magenta')
    axGeneric.plot(  get_ret_heuristic(theta)[1] , get_ret_heuristic(theta)[2]  ,color = 'red')
    axGeneric.set_xlim([0,2])
    #pdb.set_trace()
    plt.show()






def wppf(  alpha,  minMax, p=None):
    #individual_er = price_df.replace([np.inf, -np.inf], np.nan)
    # Create min value vector
    min_obj = np.array(
        [[minMax['obj2']['min'], minMax['obj1']['min'] ] ])
    max_obj = np.array(
        [[minMax['obj2']['max'], minMax['obj1']['max'] ] ] )
    d = (max_obj-min_obj ).reshape( 2,1 )
    n_d = 3
    box = d/n_d

    
    frontier_weights = []
    frontier_data = pd.DataFrame()
    S_p = np.array([])

    # Create pseudopreference vector
    args = {}
    args['n_p'] = 0
    args['alpha'] = alpha
    args['min_obj'] = min_obj
    args['max_obj'] = max_obj
    
    t_array = []
    t_fun = []
    r_array = []
    for n_p in range(1, 6):

        args['n_p'] = n_p
        alpha_min = max([-0.5 + (n_p-1)/(4*n_d), -1])
        alpha_max = min([0.5 + (n_p-1)/(4*n_d), 1])
        #pdb.set_trace()
        for alpha_1 in [i for i in np.arange(alpha_min ,alpha_max , (alpha_max-alpha_min)/2 )]: #[0]:

            # Create offset vector
            S_f = alpha_1*d
            #print (S_f)
            
            #print (n_p,box)
            #pdb.set_trace()
            S_p = -(n_p-1)*0.25*box
            

            P_v = ( np.array([[0, 0.25, 0.5, 0.75, 1]]))
            box = ( np.array(box))
            P_0 =  ( np.dot(box, P_v) )
            E = ( np.array([[1, 1, 1, 1, 1]]))

            args['S_f'] = S_f
            args['S_p'] = S_p
            args['P_0'] = P_0
            args['E'] = E
            args['bounds'] = [[0.5326,2.2532]]
            args['gamma_step'] =  1/(3*( math.sqrt(200)))
            #self.args = args
            #pdb.set_trace()
            # loop through three gamma values
            for gamma1 in np.arange(0, 1 + args['gamma_step'], args['gamma_step']):
                args['gamma1'] = gamma1
                #serialisable = pickle.dumps(args)
                serialisable = json.dumps(args, cls=NumpyEncoder)
                est =  runAggregateObjFunc.delay(serialisable)
                #est = runAggregateObjFunc(serialisable)
                
                #print (est.info)
                t_array += [est]

                #t = runAggregateObjFunc(serialisable)
                #t_array.append(t['x'])
                #t_fun += [t['fun']]


    '''res = AsyncResult(est.get,app=app)
    print (res)
    print(res.state) # 'SUCCESS'
    print(res.get()) # 7
    #print (len(t_array))
    #print (t_array)
    for t in t_array:
        print (t)
        #print(t.id) # 'SUCCESS'
        res = AsyncResult(t.id,app=app) 
        #print (res.get())
        #print(t.id) # '432890aa-4f02-437d-aaca-1999b70efe8d'
        #res = AsyncResult(t.id,app=app)
        print (res)
        r = t.get()
        r_array += r'''

    #frontier = r_array
    #pdb.set_trace()
    #print (frontier)
    #self.plot (t_array,minMax)
    #print(self.param)
    #data = readData("output.txt")
    
    x = []
    file_in = open('/home/prakash/BayesianCeleryRun/output.txt', 'r')
    for y in file_in.read().split('\n'):
        #print (y)
        
        y = str(y).replace('[','').replace(']','')
        if y!='':
            x.append(float(y))
        #print (float(y))
        #print (type(y))
        #if y.isdigit():
        
    #print (x)
    #plot(x,minMax)  
    return  None #frontier

def lineToData(line):
    return (float(line[0]) )

def readData(fileName):
    data = []
    with open(fileName) as f:
        for line in f.readlines():
            data.append(lineToData(line.split()))
    return data

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
if __name__ == "__main__":
    #p = paretoOptimiser()
    minMax = getMinMaxCaseStudy()
    #print (minMax)
    from os.path import exists
    import os

    file_exists = exists('output.txt')
    
    if file_exists:
        os.remove('output.txt')
    else:
        pass

    wppf(0.5,  minMax )
    
    ''''
    p = paretoOptimiser()
    #p.add.delay(43,4)
    t_array = []
    import time
    startTime =  time.time()
    for i in range(1000):
        #p.runOpt()
        p.runOpt.delay()
        #t_array += [p.runOpt.delay() ]
    endTime = time.time()
    print (endTime - startTime)
        #t_array += [p.runOpt.delay() ]'''
    
    #minMax = p.getMinMaxCaseStudy()
    #ranges = [[0.5326,1.2532]]
    #rC = runCelery()
    #p.wppf(0.5,minMax,ranges)

    #add.delay(4, 4)
    #theta  = 1
    #weights = theta
    #p = paretoOptimiser()

