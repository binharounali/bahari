#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:05:08 2020

@author: HarounJr
"""

import pandas as pd
import main as main
import files as files
import warnings
import sys
warnings.filterwarnings("ignore")

def activate(act):
    #User Inputs from Form
    inputs = pd.read_excel('Inputform.xlsx', 'Form')
    
    #No of settings 
    no_loop = inputs['Selected'].count()
    
    #Opening the results Files
    result_file = files.openfile(act)
    
    #Activating the jobs in cluster
    for i in range(no_loop):
        if inputs.at[0,'Setting'] == act and inputs.at[0,'Selected'] == "Yes":
            if __name__ == "__main__":
                main.main(inputs,result_file)
                print("Status: Sucessfull for Setting No:",act)
        inputs.drop(0,inplace=True)
        inputs = inputs.reset_index(drop=True)
                
if __name__ == '__main__':
  row=int(sys.argv[1])
  activate(row)