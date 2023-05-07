import datetime
import random

import pandas as pd
import gym
import numpy as np
import plotly.figure_factory as ff
import openpyxl
import operative_functions as asf
import globals

class env_va(gym.Env):
        price_energy_consumption = 0.222 #euros/KWh
        index_va = ['va09', 'va10', 'va11', 'va11', 'va12', 'va12']
        df_parameters_energy=pd.DataFrame({
                        'a': [-4335, -4335, -8081.22, -141,-6011.6, -3855.45],
                        'b': [2.1, 2.1, 4.31, 3.27, 3.83, 2.4],
                        'c': [5405.53, 5405.53, 6826.2, 5943.73, 6742.25, 901.87],
                        'd': [191.27, 191.27, 240.12, 228.9, 195.85, 292.9],
                        'e': [212.31, 212.31, 319.5, 348.29, 264.99, 238.68],
                        'f': [9.44, 9.44, 12.68, 12.16, 11.74, 8.66]}, 
                        index=index_va)
        def __init__(self, env_config=None):
############################################################################ instance variables of the class 
###############################basic variables

                self.snapshot = pd.read_excel(io='C:/Users/sergy/OneDrive/Documentos/MASTER/TFM/pruebas_sin_servidor/production_snapshot_pruebas.xlsx',
                                                sheet_name='Hoja1', header=0, names = None, index_col = None,
                                                usecols= 'A:T', engine= 'openpyxl')

                self.jobs=len(self.snapshot.axes[0]) #self jobs must reduce until 0
                
                self.max_time_steps=1000 #maximum number of time steps in an episode. It is important because
                #it limits the duration between the agent and the environment. Without this paprameter, the interaction
                #could go on indefinitely, which would not be ideal for training the agent
                self.current_time_step=0
                self.action_space = gym.spaces.Discrete(self.jobs) #you can move from processing one coil to any other
                self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.jobs)) #the observation space is the current cost each time.
                #This includes if each coil meets the plant rules, the difference of thickness...
#####################################extra variables                 
                self.winner_df = pd.DataFrame()          
                self.plants_df=self.snapshot["PLANT"].value_counts()                    #dataframe that says the number of coils that each plant has to process
                self.plants_number=len(self.plants_df.axes[0])
                self.orders_df=self.snapshot["Production_Order_NR"].value_counts()      #dataframe that says the number of different orders of coils
                self.orders_number=len(self.orders_df.axes[0])
                self.assess_costs_coil = asf.va_bid_evaluation(self.df_parameters_energy, self.snapshot,self.price_energy_consumption, self.winner_df)
                self.jid_list_2= self.assess_costs_coil.loc[:,'Coil number example'].tolist()
                self.results_2 = asf.va_result(self.assess_costs_coil, self.jid_list_2)
                print(self.results_2)
                #self.costcoils=np.random.rand(self.jobs)  #cost of the products randomly calculated and scalated between 0 and 1
                self.costcoils_df=self.results_2.loc['cost']
                self.costcoils=self.costcoils_df.to_numpy()
                self.max_cost=self.results_2.iloc[-1]['cost']
                self.coil_with_min_cost=self.results_2.iloc[0]['Coil']
                
                #print(winner_df)

                '''
                self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
                "real_obs": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.jobs, 22), dtype=float
                ),
            }
        )
        '''
        '''
        def _get_current_state_representation(self):
                self.state[:, 0] = self.legal_actions[:-1]
                return {
                "real_obs": self.state,
                "action_mask": self.legal_actions,
        }

        def get_legal_actions(self):
                return self.legal_actions
        '''
        
        def reward_scaler(self, reward):
                return reward / self.max_cost
        
        def reset_model(self):
#########################basid variables
                self.costcoils=np.random.rand(self.jobs)
                self.current_time_step=0
##########################extra variables
                self.winner_df = pd.DataFrame()
                self.plants_df = pd.DataFrame()
                self.plants_number=0
                self.orders_df = pd.DataFrame()
                self.orders_num0ber=0
                self.costs_coil=pd.DataFrame()
                self.jid_list_2=list()
                self.results_2 = pd.DataFrame()
                self.winner_df = pd.DataFrame()

        def calculate_reward(self, action):
                reward=0
                reward= -1* self.costcoils(action)
                scaled_reward= self.reward_scaler(action)

                return scaled_reward
        
        def step(self, action):
                #update the snapshot and winner_dataframe. We drop the coil with min cost and we added to the winner dataframe
                self.winner_df = self.results_2.loc[0,:]
                self.all_winner_df = pd.concat([self.all_winner_df,self.winner_df])              
                self.all_winner_df = self.all_winner_df.reset_index(drop=True)
                self.snapshot.drop(self.snapshot[(self.snapshot['Coil number example']==self.coil_with_min_cost)].index, inplace =True)
                self.snapshot = self.snapshot.reset_index(drop=True)

                #update current time step
                self.current_time_step+=1
                #calculate reward
                reward= self.calculate_reward(action)
                #update state
                self.assess_costs_coil = asf.va_bid_evaluation(self.df_parameters_energy, self.snapshot,self.price_energy_consumption, self.winner_df)
                self.jid_list_2= self.assess_costs_coil.loc[:,'Coil number example'].tolist()
                self.results_2 = asf.va_result(self.assess_costs_coil, self.jid_list_2)
                self.costcoils_df=self.results_2.loc['cost']
                self.costcoils[action]=self.costcoils_df.to_numpy()[action]                    #cost of the action
                self.state=self.costcoils
                #self.update_state(action)
                #check if the episode is done
                done=self.current_time_step>=self.max_time_steps

                #create info dictionary
                info={}

                #return observation, reward,done and info
                return self.state, reward, done, info

                '''
                
                reward = 0.0
                if action == self.jobs:
                        while self.non_assigned_jobs != 0:
                                reward -= self.increase_cost_step()
                        scaled_reward = self._reward_scaler(reward)
                        self._check_no_op()
                        return (
                        self._get_current_state_representation(),
                        scaled_reward,
                        self._is_done(),
                        {},)
                '''