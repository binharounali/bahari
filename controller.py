#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:54:40 2020

@author: HarounJr
"""
import pandas as pd
import main as main
import files as files
import warnings
warnings.filterwarnings("ignore")

#User Inputs from Form
inputs = pd.read_excel('Inputform.xlsx', 'Form')

#No of settings 
no_loop = inputs['Selected'].count()

#Opening the results Files
result_file = files.openfile()

#Running the App
for i in range(no_loop):
    if inputs.at[0,'Selected'] == "Yes":
        if __name__ == "__main__":
            main.main(inputs,result_file)
            print("Status: Sucessfull for Setting No:",i+1)
    inputs.drop(0,inplace=True)
    inputs = inputs.reset_index(drop=True)
            