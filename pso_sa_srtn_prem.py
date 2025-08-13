import pandas as pd
import numpy as np
import random
import math

data = pd.read_csv('test.csv')

data = data.sort_values(by='ArrivalTime')

process_ids = data['ProcessID'].values
arrival_times = data['ArrivalTime'].values
burst_times = data['BurstTime'].values

# num_particles = 10
# max_iter = 100
# w = 0.5  # inertia weight
# c1 = 1.5  # cognitive weight
# c2 = 1.5  # social weight
# max_velocity = 2

# Define PSO parameters
# num_particles = 6
# max_iter = 10
# w = 0.8  # inertia weight
# c1 = 1.2  # cognitive weight
# c2 = 1.2  # social weight
# max_velocity = 4

num_particles = 6
max_iter = 10
w = 0.8  # inertiaweight
c1 = 1.2  # cognitiveweight
c2 = 1.8  # socialweight
max_velocity = 4

#SRJF fitness
def srjf_fitness(solution):
    current_time = 0
    sequence = []
    tat = 0
    rt = {}
    wt = {}
    process_stats = {}  #TAT, RT, and WT for each process
    
    # Sort
    sorted_processes = [pid for _, pid in sorted(zip(solution, process_ids))]
    
    for pid in sorted_processes:
        process_index = np.where(process_ids == pid)[0][0]
        current_time = max(current_time, arrival_times[process_index])
        
        for other_pid in sorted_processes:
            if other_pid != pid:
                other_process_index = np.where(process_ids == other_pid)[0][0]
                if arrival_times[other_process_index] <= current_time and burst_times[other_process_index] < burst_times[process_index]:
                    sequence.append(other_pid)
        
                    if other_pid not in rt:
                        rt[other_pid] = current_time - arrival_times[other_process_index]
            
                    tat += (current_time - arrival_times[other_process_index])
                    current_time = arrival_times[other_process_index]
                    wt[other_pid] = max(0, tat - arrival_times[other_process_index] - burst_times[other_process_index])  # Ensure WT is non-negative
                    process_stats[other_pid] = {'TAT': tat, 'RT': rt[other_pid], 'WT': wt[other_pid], 'ArrivalTime': arrival_times[other_process_index], 'BurstTime': burst_times[other_process_index]}
                    burst_times[other_process_index] -= (current_time - arrival_times[other_process_index])

       
        if pid not in rt:
            rt[pid] = current_time - arrival_times[process_index]
        
        tat += current_time - arrival_times[process_index] + burst_times[process_index]
        
        current_time += burst_times[process_index]
        
        wt[pid] = max(0, tat - arrival_times[process_index] - burst_times[process_index])  # Ensure WT is non-negative
        
        sequence.append(pid)
        
        process_stats[pid] = {'TAT': tat, 'RT': rt[pid], 'WT': wt[pid], 'ArrivalTime': arrival_times[process_index], 'BurstTime': burst_times[process_index]}
        
    return tat, rt, wt, process_stats, sequence

#SA
def simulated_annealing(initial_solution):
    current_solution = initial_solution
    best_solution = current_solution
    current_fitness, _, _, _, _ = srjf_fitness(current_solution)
    best_fitness = current_fitness

    # temperature = 10
    # cooling_rate = 0.95

    temperature = 50
    cooling_rate = 0.95

    while temperature > 1:
        new_solution = current_solution.copy()
        #perturbation to the current solution - swap basically
        idx1, idx2 = np.random.choice(range(len(new_solution)), size=2, replace=False)
        new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]

        new_fitness, _, _, _, _ = srjf_fitness(new_solution)
        delta_fitness = new_fitness - current_fitness

        # If the new solution is better, accept
        if delta_fitness < 0:
            current_solution = new_solution
            current_fitness = new_fitness
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
        else:
            # If the new solution is worse, accept with a probability based on the temperature
            if random.random() < math.exp(-delta_fitness / temperature):
                current_solution = new_solution
                current_fitness = new_fitness

        # Decrease temperature
        temperature *= cooling_rate

    return best_solution

# PSO 
def pso(srjf_fitness):
    particles = np.random.rand(num_particles, len(process_ids))
    velocities = np.zeros((num_particles, len(process_ids)))

    global_best_fitness = float('inf')
    global_best_solution = None

    for iteration in range(max_iter):
        for i in range(num_particles):

            fitness, _, _, _, _ = srjf_fitness(particles[i])

            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_solution = particles[i].copy()

    
            velocities[i] = (w * velocities[i]) + (c1 * random.random() * (global_best_solution - particles[i])) + (c2 * random.random() * (global_best_solution - particles[i]))

            velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

            particles[i] += velocities[i]

    return global_best_solution

initial_solution = pso(srjf_fitness)

final_solution = simulated_annealing(initial_solution)

best_fitness, _, _, process_stats, best_sequence = srjf_fitness(final_solution)

print("Best Sequence:", best_sequence)
print("GLOPS - Total Turnaround Time (TAT):", best_fitness)

total_rt = sum(stat['RT'] for stat in process_stats.values())
total_wt = sum(stat['WT'] for stat in process_stats.values())

print("GLOPS - Total Response Time (RT):", total_rt)
print("GLOPS - Total Waiting Time (WT):", total_wt)




sequence_data = pd.DataFrame(process_stats).T
sequence_data.index.name = 'ProcessID'
sequence_data.reset_index(inplace=True)

sequence_data.to_csv('pso_sa_srtn_prem.csv', index=False, columns=['ProcessID', 'ArrivalTime', 'BurstTime', 'TAT', 'RT', 'WT'])
