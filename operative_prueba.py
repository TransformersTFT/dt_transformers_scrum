import os, datetime, subprocess  #, pdb
import math, json, time, statistics  #, globals
import numpy as np
import pandas as pd
from datetime import timedelta,date
from random import random
from random import randrange
from spade.message import Message
import gym



def production_cost():
    z = 985
    m = 21000
    n = 0.3
    cost = float(z * 4 + m * 0.05 + n * 2)
    return cost

def transport_cost(to):
    costes_df = pd.DataFrame()
    costes_df['From'] = ['NWW1', 'NWW1', 'NWW1','NWW1','NWW1','NWW3','NWW3','NWW3','NWW3','NWW3','NWW4','NWW4','NWW4','NWW4','NWW4']
    costes_df['CrossTransport'] = [24.6, 24.6, 0, 0, 55.6, 74.8, 74.8, 50.2, 50.2, 32.3, 71.5, 71.5, 46.9,46.9, 0]
    costes_df['Supply'] = [24.6, 24.6, 21.1, 21.1, 5.7, 24.6, 24.6, 21.1, 21.1, 5.7, 24.6, 24.6, 21.1, 21.1, 5.7]
    costes_df['To'] = ['va08', 'va09', 'va10','va11','va12','va08','va09','va10','va11','va12','va08','va09','va10','va11','va12']
    costes_df = costes_df.loc[costes_df['To'] == to]
    costes_df = costes_df.reset_index(drop=True)
    return costes_df

  #energy consumption of each coil evaluation-Sergio 25 feb####################################

def va_rules(coil_msgs_df, va_data_df):
    to= va_data_df.loc[0,'id'].split('@')[0]
    if to == 'va09':
        if abs((coil_msgs_df.loc[i,'ancho'] - va_data_df.loc[0,'coil_width'])) > 120:
            accept = 0
        else:
            accept = 1
    return accept

def energy_cost(df_parameters_energy, va_data_df, process_df, price_energy_consumption):
    to= va_data_df.loc[0,'id'].split('@')[0]
    if to == 'va09' or to == 'va10' or to == 'va11' or to == 'va12':
        a = df_parameters_energy.loc[to,'a']
        b = df_parameters_energy.loc[to,'b']
        c = df_parameters_energy.loc[to,'c']
        d = df_parameters_energy.loc[to,'d']
        e = df_parameters_energy.loc[to,'e']
        f = df_parameters_energy.loc[to,'f']
        melting_code = df_parameters_energy.loc[to,'melting_code']
        power = a + va_data_df.loc[0,'coil_width'] * b + va_data_df.loc[0, 'coil_thickness'] * \
            c + process_df.loc[0, 'setup_speed'] * f
    else:
        power=0

#   Power [kw] =     a + Width-Out (VA) * b + Final Thickness * c + Tin Layer Up * d + Tin Layer Down * e + Speed (VA) * f [kw]  
    production_time = process_df.loc[0,'processing_time'] /60                                          # Production time of the coil at the finishing line (min)
    energy_demand = power * production_time / 60                                                 # KWh
    energy_cost= energy_demand* price_energy_consumption                                               
    return energy_cost
######################
def va_bid_evaluation(coil_msgs_df, va_data_df, price_energy_consumption, df_parameters_energy, process_df):
    key = []
    transport_cost_df = transport_cost(va_data_df.loc[0,'id'].split('@')[0])
    # energy_cost=energy_cost(df_parameters_energy, va_data_df, process_df, price_energy_consumption)
    #parametros: va_df: agent_name, agent_full_name,ancho,espesor,largo,param_f,sgrade,location,ordr
    #parametros coil_msgs_df:  agent_type', 'id', 'coil_jid', 'bid', 'minimum_price','difference', 'ancho', 'largo','espesor', 'ship_date','budget_remaining'

    for i in range(transport_cost_df.shape[0]):
        m = transport_cost_df.loc[i, 'CrossTransport']
        n = transport_cost_df.loc[i, 'Supply']
        key.append(n+m)
    transport_cost_df['transport_cost'] = key
    transport_cost_df = transport_cost_df.loc[:, ['From', 'To', 'transport_cost']]
    production= production_cost()
   
    coil_msgs_df.at[0, 'production_cost'] = production
    print (coil_msgs_df.loc[0, 'production_cost'])
    ###
    energy =  energy_cost(df_parameters_energy, va_data_df, process_df, price_energy_consumption) 
        #we introduce the energy consumption of each coil in the coil_msgs_df-Sergio 25 feb####################################
    #
    print (energy)
    coil_msgs_df.loc[0, 'transport_cost'] = transport_cost_df.loc[0,'transport_cost']
    
    m = coil_msgs_df.loc[0, 'production_cost']
    n = coil_msgs_df.loc[0, 'transport_cost']
    #j = coil_msgs_df.loc[i, 'energy_cost']                                                     # Sergio 25 feb
    coil_msgs_df.loc[0, 'minimum_price'] = m + n
    coil_msgs_df.loc [0, 'minimum_price']=coil_msgs_df.loc[0, 'minimum_price'] + energy
    results= coil_msgs_df.loc[0, 'minimum_price']
    return results