# Import the W&B Python Library and log into W&B
import wandb

import os
import subprocess

#import csv

wandb.login()


project_name = "try_bayes"

default_values = {
    #"NUM_EPOCHS": 2000, # will be overwritten by num_runsteps # 2000 == 2e3  # 2e5 == 200,000
    #"NUM_RUNSTEPS": 5e4, # 5e4== 50,000
    "NUM_RUNSTEPS": 4e7, # 2e7== 20,000,000  
    # note! NUM_RUNSTEPS is batch size * num epochs.  
    # NUM_EPOCHS here will be overwritten by NUM_RUNSTEPS / BATCH_SIZE



    "DECAY": 0.8,
    "LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 20,
    "STEP_BATCH": True,

    "NUM_TUBES"  : 4,
    "NUM_COLORS" : 2,

    #"TRAIN_LEVEL_TYPE":'random',
    #"TRAIN_LEVEL_TYPE":'one_or_two',
    #"TRAIN_LEVEL_TYPE":'load_demo_one_move_rand',
    "TRAIN_LEVEL_TYPE":'scramble',
    #"TRAIN_LEVEL_TYPE":'scram_ceil',
    "TRAIN_LEVEL_PARAM": 10,
    
    "SQUARED_OUTPUT" : True,
    "WRITE_LOG" : True,
    "EXHAUSTIVE" : False,


    "loss_function" : 'MSE',
    #"loss_function" : 'MAE', # "nn.L1Loss()",  # MAE  mean absolute error
    #loss_function : 'Huber' # nn.SmoothL1Loss()  #huber 

    "DYN_LEARNING_RATE" : False,
    #"STEP_LEARN_RATE"   : True,
    "STEP_LEARN_RATE"   : False,


    "WIN_REWARD" : 100,
    "NN_SHAPE" : ["I", "2I", "2I", "2I" ,"O"]   
}

base_path = '../py/wandb_ball_runs/try_bayes/'

# create a directory for this run, and write the config to a file
# then run the model and save the model to the directory
def objective(config):

    script_path = 'deepQsolver.py'
    #id_number = wandb.run.id    # I donno, maybe this is a good id number to use?


    num_dirs = len(next(os.walk(base_path))[1])
    con_num = num_dirs + 100
    dir_name = f'sweep_con_{con_num}'
    os.makedirs(base_path + dir_name, exist_ok=False)
    config['con_num'] = con_num
    #config['dir_name'] = dir_name
    dir_path = base_path + dir_name

    print("in obj");
    print(f"{config=}")
    #print(f"{config.NN_SHAPE=}")
    print(f"\n{config.DECAY=}\n")
    ##print(f"\n{config.NUM_EPOCHS=}\n")
    print(f"\n{config.NUM_RUNSTEPS=}\n")
    print(f"\n{config.BATCH_SIZE=}\n")

    NUM_EPOCHS = int(config.NUM_RUNSTEPS / config.BATCH_SIZE)
    config.NUM_EPOCHS = NUM_EPOCHS
    print(f"\n{config.NUM_EPOCHS=}\n")




    # Write the configuration to a Python file
    with open(os.path.join(base_path + dir_name, 'config.py'), 'w') as f:
        for key, value in config.items():
            if isinstance(value, str):
                f.write(f'{key.upper()} = \"{value}\"\n')
            else:
                f.write(f'{key.upper()} = {value}\n')


    ## Write the configuration to a CSV file
    #with open(os.path.join(base_path + dir_name, 'config.csv'), 'w') as f:
    #    writer = csv.writer(f)
    #    for key, value in config.items():
    #        writer.writerow([key, value])

    print('run the deepQsolver !');
    subprocess.call(['python3', script_path, str(con_num), 'sweep'])
    # wait for it to run..
    print('done training, now load the model and run the test levels')
    import test_models_lib as lib
    model_path = f'{dir_path}/model.pth'
    print(f"{model_path=}")

    print('run the test levels !');    
    res = lib.run_2x4_tests(model_path)
    print(f"{res=}")
    print(f"{res[0]=}")
    print(f"{res[1]=}")


    score = res[0]

    return score


def main():
    global project_name

    wandb.init(project=project_name)
    score = objective(wandb.config)

    wandb.log({"score": score})


# 2: Define the search space
sweep_configuration = {
    #"method": "random",
    "method": "bayes",

    # metric is not needed for random search but here for now anyway
    "metric": {"goal": "maximize", "name": "score"},

    "parameters": {

        #"DECAY": 0.8,
        "DECAY": {"values": [0.8, 0.85, 0.9, 0.95]},
        #"LEARNING_RATE": 1e-3, # 1e-3 == 0.001
        #"BATCH_SIZE": 20,   
        "BATCH_SIZE": {"values": [16,25,32,40,64,90]},

        "LEARNING_RATE" : {"values": [1e-3, 5e-3, 5e-4,]}, 

        "NN_SHAPE" : {"values":[
            ["I",  "I",  "I",  "I" ,"O"],
            ["I",  "I", "4I",  "I" ,"O"],
            ["I", "2I", "4I", "2I" ,"O"],
            ["I", "3I", "3I", "3I" ,"O"],
            ["I",  "I",  "I", "4I" ,"O"],
            ["I",  "I", "3I",  "I" ,"O"],
            ["I", "3I",  "I",  "I" ,"O"],
            ["I", "4I",  "I",  "I" ,"O"],
            ["I", "4I", "4I",  "I" ,"O"],
            ["I",  "I", "4I", "4I" ,"O"],
            ["I", "4I", "4I", "4I" ,"O"],
            ["I", "2I", "4I",  "I" ,"O"]
        ]},  

    },
}
   


#print(f"{sweep_configuration=}")
for key in default_values:
    #if key not in sweep_configuration['parameters']:
    #    sweep_configuration['parameters'][key] = {'value': default_values[key]}
    # this next line does same as above 2 lines. Just practicing my pythonic wanking
    sweep_configuration['parameters'].setdefault(key, {'value': default_values[key]})

# set fixed values like this:
#parameters_dict.update({
#    'epochs': {'value': 1}  
#    })

#print(f"{sweep_configuration=}")


# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
print(f"{sweep_id=}")

wandb.agent(sweep_id, function=main, count=10)
