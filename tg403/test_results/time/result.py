import os
import pandas as pd


def report(file):
    df_tmp = pd.read_csv(file)
    metric = ['precision', 'recall', 'f1', 'accuracy', 'tau']
    
    df = df_tmp[metric]
    
    avg = df.mean(axis = 0).round(3)
    se = df.sem(axis = 0).round(3)
    
    print('accuracy = ', avg['accuracy'], '(', se['accuracy'], ')',
          '\nprecision = ', avg['precision'], '(', se['precision'], ')', 
          '\nrecall = ', avg['recall'], '(', se['recall'], ')',
          '\nf1 = ', avg['f1'], '(', se['f1'], ')',
          '\ntau = ', avg['tau'], '(', se['tau'], ')'
          )
    
    

file_list = os.listdir()
report('mgl_mlp.csv')



