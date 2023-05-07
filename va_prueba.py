import time, datetime, sys, os, argparse,json, re
import socket, globals, random, pdb
import pandas as pd
import numpy as np
import operative_prueba as asf
from spade import quit_spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.template import Template
from spade.message import Message
import random
import gym
import gym_jobshop
env = gym.make('jobshop-v0')
class VA:
    global coil_msgs_df, va_data_df, process_df,\
           df_parameters_energy, price_energy_consumption, all_winner_df, winner_df

    




      ##################################################################################### Dataframe parameters excel 25 feb Sergio
    index_va = ['va09', 'va10', 'va11', 'va11', 'va12', 'va12']
    df_parameters_energy=pd.DataFrame({'melting_code': ['*', '*', '1', '2','1','2'],
              'a': [-4335, -4335, -8081.22, -141,-6011.6, -3855.45],
              'b': [2.1, 2.1, 4.31, 3.27, 3.83, 2.4],
              'c': [5405.53, 5405.53, 6826.2, 5943.73, 6742.25, 901.87],
              'd': [191.27, 191.27, 240.12, 228.9, 195.85, 292.9],
              'e': [212.31, 212.31, 319.5, 348.29, 264.99, 238.68],
              'f': [9.44, 9.44, 12.68, 12.16, 11.74, 8.66]}, 
              index=index_va)
    #########################################################################################                
    price_energy_consumption = 0.222 #euros/KWh   Sergio 25 feb
    all_winner_df=pd.DataFrame()
    all_winner_df_2=pd.DataFrame()
    process_df = pd.DataFrame([], columns=['fab_start', 'processing_time', \
                'setup_speed', 'coil_width','coil_length', 'coil_thickness'])
    i=0
    j=0
    while i < 6:
      coil=random.randrange(0,6,1)
      price=random.randrange(0,200,1)
      bid=random.randrange(0,400,1)
      difference=random.randrange(0,200,1)
      budget=random.randrange(0,250,1)
      couterbid=random.randrange(0,550,1)
      profit=random.randrange(0,100,1)
      winner_df=pd.DataFrame({'Coil' : [coil],
                                  'Minimum_price' : [price],
                                  'Bid' : [bid],
                                  'Difference' : [12],
                                  'Budget_remaining' : [difference],
                                  'Counterbid' : [budget], 
                                  'Profit' : [profit]},
                                columns = ['Coil', 'Minimum_price', 'Bid', 'Budget_remaining', 'Counterbid','Profit'])
      '''
      if i == 0:
        all_winner_df = pd.concat([all_winner_df,winner_df])
        all_winner_df = all_winner_df.reset_index(drop=True)
      else :
         for j in all_winner_df.shape[0] :
        #for j in len(all_winner_df.index) :
          if winner_df[i,coil] != all_winner_df[j, coil]:
            all_winner_df = pd.concat([all_winner_df,winner_df])
            all_winner_df = all_winner_df.reset_index(drop=True)
          j+=1
          '''
      all_winner_df= pd.concat([all_winner_df,winner_df])
      all_winner_df = all_winner_df.reset_index(drop=True)
      all_winner_df_2= all_winner_df.drop_duplicates(subset=["Coil"],keep= False)
      all_winner_df_2 = all_winner_df_2.reset_index(drop=True)
      size=all_winner_df.shape[0]
      #print("todas las ofertas")
      #print(all_winner_df)
      #print("sin repetir")
      #print(all_winner_df_2)
      i+=1
    
    process_df.at[0,'setup_speed'] = 0.25 # Normal speed 15000 mm/min in m/s
    process_df = process_df.reset_index(drop=True)
    processing_time = 100
    process_df['processing_time'].iloc[-1] = processing_time
     
    process_df['fab_start'].iloc[-1] = 10
    va_data_df= pd.DataFrame()
    coil_msgs_df= pd.DataFrame()
     #parametros: va_df: agent_name, agent_full_name,ancho,espesor,largo,param_f,sgrade,location,ordr
    va_data_df.at[0, 'coil_width'] = 980
    va_data_df.at[0,'coil_thickness'] = 0.5
    process_df.at[0, 'setup_speed'] = 0.3
    process_df.at[0,'processing_time'] =75000
    va_data_df.at[0, 'auction_level'] = 1
    va_data_df.at[0,'id'] = "va10@etsii.com"
    va_data_df.at[0, 'bid_status'] = 'bid'
    va_data_df['accumulated_profit'] = 0
    to= va_data_df.loc[0,'id'].split('@')[0]
    
         
    bid_coil = asf.va_bid_evaluation(coil_msgs_df, va_data_df, price_energy_consumption, df_parameters_energy, process_df)

    
    print(bid_coil)
