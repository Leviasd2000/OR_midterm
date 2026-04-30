import pandas as pd
import gurobipy as gp
import math
import os

from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
from gurobipy import GRB
from pprint import pprint


os.environ["GRB_LICENSE_FILE"] = (
    r"C:\Users\Levi\gurobi.lic"
)

start_time = datetime(2023, 1, 1, 0, 0)

def datetime_to_minutes(time):
    hours = datetime.strptime(time, "%Y/%m/%d %H:%M")
    return int((hours - start_time).total_seconds() // 1800)

def parse_text(file_name):
    current_dir = os.getcwd() 

    file_path = os.path.join(current_dir, "OR114-2_midterm_data", file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_name} not found in {current_dir}")
    
    with open(file_path, 'r') as f:
        content = f.read().strip()
        
    blocks = [block.strip() for block in content.split('==========')]
    
    # cut into five different blocks
    
    # ---
    # block 1: total number of every index set
    # ---
    
    start_time = datetime(2023, 1, 1, 0, 0)
    lines_1 = blocks[0].split('\n')
    keys = lines_1[0].split(',')
    vals = lines_1[1].split(',')
    params = {k: int(v) for k, v in zip(keys, vals)}
    
    # ---
    # block 2: car state
    # ---
            
    lines_2 = blocks[1].split('\n')
    car_id = []
    car_level = []
    car_init = []
    for line in lines_2[1:]:
        id, level ,station = line.split(',')
        car_id.append(int(id))
        car_level.append(int(level))
        car_init.append(int(station))
     
    # ---
    # block 3: car rates
    # ---
    
    lines_3 = blocks[2].split('\n')
    car_level_rate = []
    for line in lines_3[1:]:
        level, rate = line.split(',')
        car_level_rate.append(float(rate))
        
    # ---
    # block 4: order info
    # ---   
    
    lines_4 = blocks[3].split('\n')
    order_level = []
    order_pickup = []
    order_return = []
    order_pickup_time = []
    order_return_time = []
    order_hours = []
    
    for line in lines_4[1:]:
        id, level, pickup, return_, pickup_time, return_time = line.split(',')
        order_level.append(int(level))
        order_pickup.append(int(pickup))
        order_return.append(int(return_))
        order_pickup_time.append(int(datetime_to_minutes(pickup_time)))
        order_return_time.append(int(datetime_to_minutes(return_time)))
        order_hours.append((datetime_to_minutes(return_time) - datetime_to_minutes(pickup_time))/2)

    # ---
    # block 5: Transfer time
    # ---   
    
    lines_5 = blocks[4].split('\n')
    transfer_time = {}
    for line in lines_5[1:]:
        from_, to, time = line.split(',')
        transfer_time[(int(from_), int(to))] = int(time)
    
    data = {
        "params": params,
        "car_info": {
            "id": car_id,
            "level": car_level,
            "init": car_init
        },
        "car_rates": car_level_rate,
        "orders": {
            "level": order_level,
            "pickup": order_pickup,
            "return": order_return,
            "pickup_time": order_pickup_time,
            "return_time": order_return_time,
            "hours": order_hours
        },
        "transfer_time": transfer_time
    }
    return data
        
def problem1(file_name):
    data = parse_text(file_name)
    
    # Extracting data
    num_cars = data["params"]["n_C"]
    num_orders = data["params"]["n_K"]
    num_stations = data["params"]["n_S"]
    
    car_id = data["car_info"]["id"]
    car_level = data["car_info"]["level"]
    car_init = data["car_info"]["init"]
    
    car_level_rate = data["car_rates"]
    
    order_level = data["orders"]["level"]
    order_pickup = data["orders"]["pickup"]
    order_return = data["orders"]["return"]
    order_pickup_time = data["orders"]["pickup_time"]
    order_return_time = data["orders"]["return_time"]
    order_hours = data["orders"]["hours"]
    
    transfer_time = data["transfer_time"]
    
    # Create Variables
    
    C = range(1, num_cars + 1)
    O = range(1, num_orders + 1)
    O_ = range(0, num_orders + 1) # add dummy order 0
    V = 11260000
    B = data["params"]["B"] / 30
    
    transfer_matrix = [[0] * (num_stations + 1) for _ in range(num_stations + 1)]
    
    for (from_, to), time in transfer_time.items():
        transfer_matrix[from_][to] = time / 30
            
    # Create Model
    m = gp.Model("Car_Rental_Problem")
    # Create decision variables
    
    x = m.addVars(O, vtype=GRB.BINARY, name="x")
    y = m.addVars(C, O, vtype=GRB.BINARY, name="y")
    u = m.addVars(O, vtype=GRB.BINARY, name="u")
    z = m.addVars(C, O_, O_, vtype=GRB.BINARY, name="z")
    
    # Objective Function
    m.setObjective(gp.quicksum(3 * car_level_rate[order_level[o-1]-1] * order_hours[o-1] * x[o] - 2 * car_level_rate[order_level[o-1]-1] * order_hours[o-1] for o in O), GRB.MAXIMIZE)
    # Constraints
    m.addConstrs((x[o] == gp.quicksum(y[c, o] for c in C) for o in O), name="One_Car_Per_Order")
    
    m.addConstrs((gp.quicksum(car_level[c-1] * y[c, o] for c in C) == order_level[o-1] * x[o] + u[o] for o in O), name="Car_Upgrade_Level")
    
    m.addConstrs((z.sum(c, 0, '*') <= 1 for c in C), name="Start_Flow")
    
    m.addConstrs((z.sum(c, '*', 0) <= 1 for c in C), name="End_Flow")
    
    m.addConstrs(
        (gp.quicksum(z[c, j, k] for j in O_ if j != k) == y[c, k] 
         for c in C for k in O), 
        name="In_Flow"
    )
    
    m.addConstrs(
        (gp.quicksum(z[c, j, k] for k in O if k != j) <= y[c, j] 
         for c in C for j in O), 
        name="Out_Flow"
    )   
    
    m.addConstrs(
        (order_pickup_time[k-1] - order_return_time[j-1] + V * (1 - z[c, j, k]) 
         >= 9 + transfer_matrix[order_return[j-1]][order_pickup[k-1]]
         for c in C for j in O for k in O if j != k), 
        name="Time_Transition"
    )
    
    m.addConstrs(
        (order_pickup_time[k-1] + V * (1 - z[c, 0, k]) 
         >= 1 +transfer_matrix[car_init[c-1]][order_pickup[k-1]]
         for c in C for k in O if car_init[c-1] != order_pickup[k-1]), 
        name="Time_First_Task_transition"
    )
    
    m.addConstrs(
        (order_pickup_time[k-1] + V * (1 - z[c, 0, k]) 
         >= transfer_matrix[car_init[c-1]][order_pickup[k-1]]
         for c in C for k in O if car_init[c-1] == order_pickup[k-1]), 
        name="Time_First_Task"
    )
    
    m.addConstr(
        gp.quicksum(
            transfer_matrix[car_init[c-1]][order_pickup[k-1]] * z[c, 0, k]
            for c in C for k in O
        ) +
        gp.quicksum(
            transfer_matrix[order_return[j-1]][order_pickup[k-1]] * z[c, j, k]
            for c in C for j in O for k in O  if j != k
        ) <= B,
        name="Total_Budget"
    )
    
    m.optimize()
    # First, ensure the model actually found a mathematical solution
    if m.status == GRB.OPTIMAL:
        print("\n=== INVENTORY TRACKER ===")
        
data_set = ["instance01.txt", "instance02.txt", "instance03.txt", "instance04.txt", "instance05.txt"]

for file_name in data_set:    
    print(f"\n>>> 開始求解 {file_name} ...")
    problem1(file_name)
        

    

    


    
    