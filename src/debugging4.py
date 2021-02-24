'''
.. module:: Safe RL

.. moduleauthor:: Ahmet Semi ASARKAYA <asark001@umn.edu.edu>

'''

import logging, sys
import StringIO
import pdb, os, copy, math
import time, timeit
import operator
import csv

import os
import scipy as sp # get adjacency
import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib import colors

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

import twtl
import write_files

from collections import Counter
from create_environment import create_ts, update_obs_mat, update_adj_mat_3D,\
								create_input_file, update_adj_mat_3D

from dfa import DFAType
from synthesis import expand_duration_ts, compute_control_policy, ts_times_fsa,\
                      verify, compute_energy                  
from geometric_funcs import check_intersect, downwash_check
from write_files import write_to_land_file, write_to_csv_iter, write_to_csv,\
                        write_to_iter_file, write_to_control_policy_file
from learning import learn_deadlines
from lomap import Ts


def prep_for_learning(ep_len, m, n, h, init_states, obstacles, pick_up_state, delivery_state, rewards, rew_val, custom_flag, custom_task):
	# Create the environment and get the TS #
	ts_start_time = timeit.default_timer()
	disc = 1
	TS, obs_mat, state_mat = create_ts(m,n,h)	
	path = '../data/ts_' + str(m) + 'x' + str(n) + 'x' + str(h) + '_1Ag_1.txt'
	paths = [path]
	bases = {init_states[0]: 'Base1'}
	obs_mat = update_obs_mat(obs_mat, state_mat, m, obstacles, init_states[0])
	TS      = update_adj_mat_3D(m, n, h, TS, obs_mat)
	create_input_file(TS, state_mat, obs_mat, paths[0], bases, disc, m, n, h, 0)
	ts_file = paths
	ts_dict = Ts(directed=True, multi=False) 
	ts_dict.read_from_file(ts_file[0])
	ts = expand_duration_ts(ts_dict)
	ts_timecost =  timeit.default_timer() - ts_start_time

	# Get the DFA #
	dfa_start_time = timeit.default_timer()
	pick_up  = str(pick_up_state[0][0] * n + pick_up_state[0][1])
	delivery = str(delivery_state[0][0] * n + delivery_state[0][1])
	tf  = str(ep_len) # time bound
	if custom_flag == 1:
		phi = custom_task
	else:
		phi = '([H^1 r' + pick_up + ']^[0, 15] * [H^1 r' + delivery + ']^[0, 15])^[0, 31]'
		#phi = '([H^1 r' + pick_up + ']^[0, ' + tf + '] * [H^1 r' + delivery + ']^[0,' + tf + '])^[0, ' + tf + ']' # Construc the task according to pickup/delivery
	_, dfa_nor, bdd = twtl.translate(phi, kind=DFAType.Infinity, norm=True) # states and sim. time ex. phi = '([H^1 r47]^[0, 30] * [H^1 r31]^[0, 30])^[0, 30]' 
	dfa_timecost =  timeit.default_timer() - dfa_start_time # DFAType.Normal for normal, DFAType.Infinity for relaxed

	# Get the PA #
	pa_start_time = timeit.default_timer()
	alpha = 1
	nom_weight_dict = {}
	weight_dict = {}
	pa_or = ts_times_fsa(ts, dfa_nor) # Original pa
	edges_all = nx.get_edge_attributes(ts_dict.g,'edge_weight')
	max_edge = max(edges_all, key=edges_all.get)
	norm_factor = edges_all[max_edge]
	for pa_edge in pa_or.g.edges():
		edge = (pa_edge[0][0], pa_edge[1][0], 0)
		nom_weight_dict[pa_edge] = edges_all[edge]/norm_factor
	nx.set_edge_attributes(pa_or.g, 'edge_weight', nom_weight_dict)
	nx.set_edge_attributes(pa_or.g, 'weight', 1)
	pa = copy.deepcopy(pa_or)	      # copy the pa
	time_weight = nx.get_edge_attributes(pa.g,'weight')
	edge_weight = nx.get_edge_attributes(pa.g,'edge_weight')
	for pa_edge in pa.g.edges():
		weight_dict[pa_edge] = alpha*time_weight[pa_edge] + (1-alpha)*edge_weight[pa_edge]
	nx.set_edge_attributes(pa.g, 'new_weight', weight_dict)
	pa_timecost =  timeit.default_timer() - pa_start_time

	# Compute the energy of the states #
	energy_time = timeit.default_timer()
	compute_energy(pa)
	energy_dict = nx.get_node_attributes(pa.g,'energy')
	energy_pa    = []

	for ind in range(len(pa.g.nodes())):
		energy_pa.append(pa.g.nodes([0])[ind][1].values()[0])
	
	blocking_inds = []
	for ind in range(len(pa.g.nodes())):
		if energy_pa[ind] > 100000:
			blocking_inds.append(ind)
		
	pa_g_nodes = pa.g.nodes()
	for ind in sorted(blocking_inds, reverse=True):
		del pa_g_nodes[ind]
		del energy_pa[ind]

	# projection of pa on ts #
	init_state = [init_states[0][0] * n + init_states[0][1]]
	pa2ts = []
	for i in range(len(pa_g_nodes)):
		if pa_g_nodes[i][0] != 'Base1':
			pa2ts.append(int(pa_g_nodes[i][0].replace("r","")))
		else:
			pa2ts.append(init_state[0])
			i_s = i # Agent's initial location in pa
	energy_timecost =  timeit.default_timer() - pa_start_time

	# TS adjacency matrix and source-target
	TS_adj = TS
	TS_s   = []
	TS_t   = []
	for i in range(len(TS_adj)):
		for j in range(len(TS_adj)):
			if TS_adj[i,j] != 0:
				TS_s.append(i)
				TS_t.append(j)

	# pa adjacency matrix and source-target 
	pa_adj_st = nx.adjacency_matrix(pa.g)
	pa_adj    = pa_adj_st.todense()
	pa_s = [] # source node
	pa_t = [] # target node
	c1 = -1
	for i in range(len(pa_adj)):
		if i not in blocking_inds:
			c1 = c1 + 1
			c2 = -1
			for j in range(len(pa_adj)):
				if j not in blocking_inds:
					c2 = c2 + 1
					if pa_adj[i,j] == 1:
						pa_s.append(c1)
						pa_t.append(c2)
	
	blocking_pa_inds = []
	for ind in range(len(pa_s)):
		if pa_s[ind] in blocking_inds:
			blocking_pa_inds.append(ind)
		elif pa_t[ind] in blocking_inds:
			blocking_pa_inds.append(ind)

	#for ind in sorted(blocking_pa_inds, reverse=True):
	#	del pa_s[ind]
	#	del pa_t[ind]

    # PA rewards matrix
	rewards_ts = np.zeros(m * n)
	rewards_pa = np.zeros(len(pa2ts))
	rewards_ts_indexes = []
	for i in range(len(rewards)):
		rewards_ts_indexes.append(rewards[i][0] * n + rewards[i][1]) # rewards_ts_indexes[i] = rewards[i][0] * n + rewards[i][1]		
		rewards_ts[rewards_ts_indexes[i]] = rew_val
	
	for i in range(len(rewards_pa)):
		rewards_pa[i] = rewards_ts[pa2ts[i]]
	
	# # Display some important info
	print('##### PICK-UP and DELIVERY MISSION #####' + "\n")
	print('Initial Location  : ' + str(init_states[0]) + ' <---> Region ' + str(init_state[0]))
	print('Pick-up Location  : ' + str(pick_up_state[0]) + ' <---> Region ' + pick_up)
	print('Delivery Location : ' + str(delivery_state[0]) + ' <---> Regions ' + delivery)
	print('Reward Locations  : ' + str(rewards) + ' <---> Regions ' + str(rewards_ts_indexes) + "\n")
	print('State Matrix : ')
	print(state_mat)
	print("\n")
	print('Mission Duration  : ' + tf + ' time steps')
	print('TWTL Task : ' + phi + "\n")
	print('Computational Costst : TS created in ' + str(ts_timecost) + ' seconds')
	# print('			TS created in ' + str(ts_timecost) + ' seconds')
	print('		       DFA created in ' + str(dfa_timecost) + ' seconds')
	print('		       PA created in ' + str(pa_timecost) + ' seconds')
	print('		       Energy of PA states calculated in ' + str(energy_timecost) + ' seconds')

	return i_s, pa, pa_s, pa_t, pa2ts, energy_pa, rewards_pa, pick_up, delivery, pa_g_nodes


def get_possible_actions(energy_pa, pa2ts, pa_s, pa_t):
	# Remove blocking states and the corresponding transitionswhen the hit flag is raised
	accepting_states = []
	for ind, val in enumerate(energy_pa):   # Find the accepting states
		if val == 0:
			accepting_states.append(ind)

	diff_s2t = []
	nth_row_pa_t = []
	nth_row_pa_s = []
	for i in range(len(pa_s)):
		diff_s2t.append(pa2ts[pa_t[i]] - pa2ts[pa_s[i]])
		nth_row_pa_t.append(int(math.ceil((pa2ts[pa_t[i]]+1.0)/n)))
		nth_row_pa_s.append(int(math.ceil((pa2ts[pa_s[i]]+1.0)/n)))
	
	act_s2t = []
	act_num = [] 			                            #####################################
	for ind, diff in enumerate(diff_s2t):               #                                   #
		if diff == n - 1 and (nth_row_pa_t[ind] != nth_row_pa_s[ind]):                               #Actions and Corresponding Numbering#
			act_s2t.append('SouthWest')					#                                   #
			act_num.append(6)						 	#  --------------          -------  #
		elif diff == n:	                                #  |NW   N    NE|          |0 1 2|  #
			act_s2t.append('South')                     #  |W   Stay  E |  <-----> |3 4 5|  #
			act_num.append(7)                           #  |SW   S    SE|          |6 7 8|  #
		elif diff == n + 1:                             #  --------------          -------  #
			act_s2t.append('SouthEast')                 #                                   #
			act_num.append(8)                           #####################################
		elif diff == -1 and (nth_row_pa_t[ind] == nth_row_pa_s[ind]):
			act_s2t.append('West')
			act_num.append(3)
		elif diff == 0:
			act_s2t.append('Stay')
			act_num.append(4)
		elif diff == 1 and (nth_row_pa_t[ind] == nth_row_pa_s[ind]):
			act_s2t.append('East')
			act_num.append(5)
		elif diff == -n - 1:
			act_s2t.append('NorthWest')
			act_num.append(0)
		elif diff == -n:
			act_s2t.append('North')
			act_num.append(1)
		elif diff == -(n-1) and (nth_row_pa_t[ind] != nth_row_pa_s[ind]):
			act_s2t.append('NorthEast')
			act_num.append(2)
	#print('diff : ', diff_s2t)
	k = 0
	possible_acts = []
	possible_next_states = []
	for i, values in enumerate(Counter(pa_s).values()):
		pos_n_acts = []
		possible_n_next_states = []
		for j in range(values):
			pos_n_acts.append(act_num[k])
			possible_n_next_states.append(pa_t[k])
			k += 1
		possible_acts.append(pos_n_acts)
		possible_next_states.append(possible_n_next_states)

	for ind in sorted(accepting_states, reverse=False):
		possible_next_states.insert(ind, [])
		possible_acts.insert(ind, [])

	return possible_acts, possible_next_states, act_num


def action_uncertainity(current_act, pa_s, pa_t, act_num, agent_s):	
	indices = []
	acts = []
	for ind, val in enumerate(pa_s):
		if agent_s == val:
			indices.append(ind)
			acts.append(act_num[ind])
	possible_acts = []
	if current_act == 0: # 1,3,4		
		if 1 in acts:
			possible_acts.append(1)
		if 3 in acts:
			possible_acts.append(3)
	elif current_act == 1: # 0,2,4
		if 0 in acts:
			possible_acts.append(0)
		if 2 in acts:
			possible_acts.append(2)
	elif current_act == 2: # 1,5,4
		if 1 in acts:
			possible_acts.append(2)
		if 5 in acts:
			possible_acts.append(5)			
	elif current_act == 3: # 0,6,4
		if 0 in acts:
			possible_acts.append(0)
		if 6 in acts:
			possible_acts.append(6)
	elif current_act == 5: # 2,8,4
		if 2 in acts:
			possible_acts.append(2)
		if 8 in acts:
			possible_acts.append(8)
	elif current_act == 6: # 3,7,4
		if 3 in acts:
			possible_acts.append(3)
		if 7 in acts:
			possible_acts.append(7)
	elif current_act == 7: # 6,8,4
		if 6 in acts:
			possible_acts.append(6)
		if 8 in acts:
			possible_acts.append(8)
	else:                                      # 5,7,4
		if 5 in acts:
			possible_acts.append(5)
		if 7 in acts:
			possible_acts.append (7)

	possible_acts.append(4)	
	chosen_act = random.choice(possible_acts)
	if chosen_act == 4:
		next_state = agent_s
	else:
		inter_ind = acts.index(chosen_act)
		next_state = pa_t[indices[inter_ind]]	

	return chosen_act, next_state


def Q_Learning(Pr_des, eps_unc, N_EPISODES, SHOW_EVERY, LEARN_RATE, DISCOUNT, EPS_DECAY, epsilon, i_s, pa, energy_pa, pa2ts, pa_s, pa_t, act_num, possible_acts, possible_next_states,  pick_up, delivery, pa_g_nodes):
	# Getting the non-accepting TS states
	agent_upt = []
	for i in range(len(pa_g_nodes)):
		if pa_g_nodes[i][1] == 0 or str(pa_g_nodes[i][0]) == 'r'+str(pick_up) : # If the mission changes check here
			agent_upt.append(pa2ts[i])
		else:
			agent_upt.append([])

	QL_start_time = timeit.default_timer()

	EVERY_PATH = []
	episode_rewards = []

	# Initialize the Q - table (Between -0.01 and 0)
	pa_size = len(pa_g_nodes) 
	q_table = np.random.rand(4 * (ep_len+1),9) * 0.001 - 0.001  # of states x # of actions

	agent_s = i_s       # Initialize the agent's location
	ep_rewards = [] 
	ep_trajectories_pa = []
	hit_count = 0

	for episode in range(N_EPISODES):

		if episode > 100000:
			epsilon = 0

		hit_ref = 0
		ep_traj_pa = [agent_s] # Initialize the episode trajectory
		ep_rew     = 0         # Initialize the total episode reward
		hit        = 0         # Set default "hit flag" to 0	
		

		for t_ep in range(ep_len):
			possible_next_states_copy = copy.deepcopy(possible_next_states)
			possible_acts_copy = copy.deepcopy(possible_acts)
			# print('step : ,', t_ep)
			# print('actions before pruning : ',possible_acts_copy[agent_s])

			k_ep = ep_len - t_ep - 1# Remaning episode time
			if hit == 0:       					                                                                                    
				if energy_pa[agent_s] == 0:  # Raise the 'hit flag' if the mission is achieved 
					hit = 1                  # re-initialize the agent_s to prevent stuck
					hit_count = hit_count + 1
					agent_s = agent_upt.index(pa2ts[agent_s]) # Reinitiliaze the pa(region, 0)
					possible_next_states_copy = copy.deepcopy(possible_next_states)
					possible_acts_copy = copy.deepcopy(possible_acts)
					#break

			if hit == 0:		
				# print('actions before pruning 2: ',possible_acts_copy[agent_s])
				en_list = [energy_pa[i] for i in possible_next_states_copy[agent_s]] # Energies of the next possible states
				# print('en_list',en_list)
				not_possible_index = []
				ind_minholder = en_list.index(min(en_list))#np.argmin(np.array(en_list))
				possible_next_states_minholder = possible_next_states_copy[agent_s][ind_minholder] 
				possible_acts_minholder = possible_acts_copy[agent_s][ind_minholder]       
				for j in range(len(possible_next_states_copy[agent_s])):
					d_max   = en_list[j] + 1
					i_max   = int(math.floor((k_ep - 1 - d_max) / 2))
					thr_fun = 0
					for i in range(i_max+1):
						thr_fun = thr_fun + np.math.factorial(k_ep) / (np.math.factorial(k_ep-i) * np.math.factorial(i)) * eps_unc**i * (1-eps_unc)**(k_ep-i)
					#print(thr_fun)
					#print('\n')	
					if  thr_fun < Pr_des and i_max > 0: #energy_pa[possible_next_states_copy[agent_s][j]] > k_ep-1: # 
						not_possible_index.append(j)
					else:	
						possible_next_states_copy[agent_s] = []
						possible_acts_copy[agent_s] = []
						break					

				if len(possible_next_states_copy[agent_s]) != 0:		
					for ind in sorted(not_possible_index, reverse=True):
						del possible_next_states_copy[agent_s][ind]
						del possible_acts_copy[agent_s][ind]

				if len(possible_next_states_copy[agent_s]) == 0: # not possible_next_states_copy[agent_s]: #
					possible_next_states_copy[agent_s].append(possible_next_states_minholder)
					possible_acts_copy[agent_s].append(possible_acts_minholder)	
			
			prev_s = agent_s
			# print('energy',energy_pa[agent_s])
			# print('state : ',pa2ts[agent_s])
			# print('actions after pruning : ',possible_acts_copy[agent_s])
			if np.random.uniform() > epsilon:                              # Exploit
				possible_qs = q_table[agent_s,possible_acts_copy[agent_s]] # Possible Q values for each action
				next_ind    = np.argmax(possible_qs)                       # Pick the action with max Q value 
			else:                                                          # Explore
				next_ind  = np.random.randint(len(possible_acts_copy[agent_s])) # Picking a random action
			# Taking the action
			intended_action = possible_acts_copy[agent_s][next_ind]
			if np.random.uniform() < eps_unc:
				[chosen_act, next_state] = action_uncertainity(intended_action	, pa_s, pa_t, act_num, agent_s)
				action    = chosen_act
				s_a       = (agent_s, action)                                   # State & Action pair
				agent_s   = next_state # possible_next_states[agent_upt.index(pa2ts[agent_s])][next_ind]        # moving to next state  (s,a)
			else:
				action    = intended_action
				s_a       = (agent_s, action)                                   # State & Action pair
				agent_s   = possible_next_states_copy[agent_s][next_ind]        # moving to next state  (s,a)	      
			# print('action taken : ', action)
			# print('\n')

			ep_traj_pa.append(agent_s)
			current_q = q_table[pa2ts[prev_s] * (ep_len+1) + t_ep, intended_action]
			max_future_q = np.amax(q_table[pa2ts[agent_s] * (ep_len+1) + t_ep+1, :])                       # Find the max future q 
			rew_obs      = np.random.binomial(1, 1-rew_uncertainity) * rewards_pa[agent_s]                 # Observe the rewards of the next state
			new_q        = (1 - LEARN_RATE) * current_q + LEARN_RATE * (rew_obs + DISCOUNT * max_future_q) # Calculate the new q value
			q_table[pa2ts[prev_s] * (ep_len+1) + t_ep, intended_action] = new_q                            # Update the table
			ep_rew += rew_obs
		
		agent_s = agent_upt.index(pa2ts[agent_s]) # Reinitiliaze the pa(region, 0) 
		ep_rewards.append(ep_rew)
		ep_trajectories_pa.append(ep_traj_pa)
		epsilon = epsilon * EPS_DECAY;
		if (episode+1) % SHOW_EVERY == 0:
			avg_rewards = np.mean(ep_rewards[episode-SHOW_EVERY +1: episode])
			print('Episode # ' + str(episode+1) + ' : Epsilon=' + str(round(epsilon, 4)) + '    Avg. reward in the last ' + str(SHOW_EVERY) + ' episodes=' + str(round(avg_rewards,5)))
	
	best_episode_index = ep_rewards.index(max(ep_rewards))
	optimal_policy_pa  = ep_trajectories_pa[N_EPISODES-1]#ep_trajectories_pa[best_episode_index] # Optimal policy in pa  ep_trajectories_pa[N_EPISODES-1]#
	optimal_policy_ts  = []                                     # optimal policy in ts
	opt_pol            = []                                     # optimal policy in (m, n, h) format for visualization
	for ind, val in enumerate(optimal_policy_pa):
		optimal_policy_ts.append(pa2ts[val])
		opt_pol.append((math.floor(optimal_policy_ts[ind]/n), optimal_policy_ts[ind]%n, 0))

	print('\n Tajectory at the last episode : ' + str(optimal_policy_ts))
	QL_timecost =  timeit.default_timer() - QL_start_time
	success_ratio = 100*hit_count/N_EPISODES
	max_energy = max(energy_pa)

	print('\n Total time for Q-Learning : ' + str(QL_timecost) + ' seconds' + "\n")
	print('Action uncertainity[%] = ' + str(eps_unc*100))
	print("Desired Minimum Success Ratio[%] = " + str(100*Pr_des))
	print("Successful Mission Ratio[%] = " + str(success_ratio))
	print("Successful Missions = " + str(hit_count) + " out of " + str(N_EPISODES))
	print("Episode Length = " + str(ep_len) + "  and  Max. Energy of the System = " + str(max_energy) + "\n")

	return opt_pol


def visualization(m, n, init_states, obstacles, pick_up_state, delivery_state, rewards, opt_pol):
	# Trajectory
	x_data = np.zeros(len(opt_pol))
	y_data = np.zeros(len(opt_pol))
	for i in range(len(opt_pol)):
	    x_data[i] = opt_pol[i][1]
	    y_data[i] = opt_pol[i][0]

	# Color accordingly
	cdata = np.zeros((m, n))
	for i in range(len(obstacles)):
	   cdata[obstacles[i][0], obstacles[i][1]] = 0.75

	for i in range(len(rewards)):
	    cdata[rewards[i][0], rewards[i][1]] = 2.75

	cdata[init_states[0][0], init_states[0][1]] = 1.25
	cdata[pick_up_state[0][0], pick_up_state[0][1]] = 1.75
	cdata[delivery_state[0][0], delivery_state[0][1]] = 2.25
	cdata[rewards[0][0], rewards[0][1]] = 2.75

	cmap = colors.ListedColormap(['white', 'red', 'yellow', 'blue', 'green', 'gray'])
	bounds = [0, 0.5 ,1, 1.5, 2, 2.5, 3]
	norm = colors.BoundaryNorm(bounds, cmap.N)

	# Create figure
	fig, ts = plt.subplots()    
	fig.suptitle('Transition System')
	ts.imshow(cdata, cmap=cmap, norm=norm)

	ts.plot(x_data,y_data, linewidth=6)

	plt.tick_params(
	axis='both',       # changes apply to both axes
	which='major',     # major ticks are affected
	bottom=False,      # ticks along the bottom edge are off
	left =False,       # ticks along the left edge are off
	labelbottom=False, # labels along the bottom edge are off
	labelleft  =False) # labels along the left edge are off

	# draw gridlines
	ts.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)    
	ts.set_xticks(np.arange(0.5, n, 1));
	ts.set_yticks(np.arange(0.5, m, 1));
	plt.show()

if __name__ == '__main__':
	os.system('clear')
	start_time  = time.time()
	custom_flag = 1          # Task flag, if zero, it is a pick-up delivery mission. If 1, define a new custom task
	custom_task = '[H^0 r1]^[0, 1] * [H^0 Base1]^[0,1]' # 
	##### System Inputs for Data Prep. #####
	ep_len = 3 # Episode length
	m = 2       # of rows
	n = 2       # of columns  8
	h = 1       # height set to 1 for 2D case
	init_states    = [(1,0,0)]                                           # Specify initial states and obstacles (row,column,altitude/height)
	obstacles = []    # Indexed from (0,0,0) which is the upper left corner at ground height
	pick_up_state  = [(0,1,0)]                                         # Specify pickup state
	delivery_state = [(0,0,0)]                                         # Specify delivery state
	rewards   = [(1,1,0)]                   # Rewarded states
	rew_val   = 1                                                      # Reward value
	rew_uncertainity = 0.00

	# Call the function 'prep_for_learning' and 'get_possible_actions' to get required parameters for learning #
	prep_start_time = timeit.default_timer()
	[i_s, pa, pa_s, pa_t, pa2ts, energy_pa, rewards_pa, pick_up, delivery, pa_g_nodes] = prep_for_learning(ep_len, m, n, h, init_states, obstacles, pick_up_state, delivery_state, rewards, rew_val, custom_flag, custom_task)
	[possible_acts, possible_next_states, act_num] = get_possible_actions(energy_pa, pa2ts, pa_s, pa_t)
	prep_timecost =  timeit.default_timer() - prep_start_time
	print('Total time for data prep. : ' + str(prep_timecost) + ' seconds \n')
	# print(pa.g.nodes())

	# print('possibble acts:',possible_acts)
	# print('pa2ts : ', pa2ts)
	# print('possible_next_states:',possible_next_states)

	# print('pa_s :   ', pa_s)
	# print('pa_t :   ', pa_t)
	# print('act num :', act_num)

	# for i in range(len(pa_s)):
	# 	print('pa_s :   ', pa2ts[pa_s[i]])
	# 	print('pa_t :   ', pa2ts[pa_t[i]])
	# 	print('act num :', act_num[i])
	# 	print('\n')

	##### System Inputs for Q-Learning #####
	N_EPISODES = 150000      # of episodes
	SHOW_EVERY = 10000        # Print out the info at every ... episode
	LEARN_RATE = 0.1
	DISCOUNT   = 0.95
	EPS_DECAY  = 1
	epsilon    = 0.1
	eps_unc    = 0.1000 # Uncertainity in actions
	Pr_des     = 0.700


	# Check possible minimum threshold
	d_max   = max(energy_pa)
	i_max   = int(math.floor((ep_len - 1 - d_max) / 2))
	thr_fun = 0
	for i in range(i_max+1):
		thr_fun = thr_fun + np.math.factorial(ep_len) / (np.math.factorial(ep_len-i) * np.math.factorial(i)) * eps_unc**i * (1-eps_unc)**(ep_len-i)
	print('Maximum threshold that can be put ' + str(thr_fun))	
	if thr_fun < Pr_des:
		print('Please set a smaller desired probability threshold than ' + str(thr_fun))
	else:
		# Call the Q_Learning Function
		opt_pol = Q_Learning(Pr_des, eps_unc, N_EPISODES, SHOW_EVERY, LEARN_RATE, DISCOUNT, EPS_DECAY, epsilon, i_s, pa, energy_pa, pa2ts, pa_s, pa_t, act_num, possible_acts, possible_next_states, pick_up, delivery, pa_g_nodes)
		print("Total run time : %s seconds" % (time.time() - start_time))

		##### Visualize ######
		visualization(m, n, init_states, obstacles, pick_up_state, delivery_state, rewards, opt_pol)

