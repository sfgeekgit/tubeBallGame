# Import the W&B Python Library and log into W&B
import wandb

wandb.login()


# to do, these are not itteralbe. so... is there a way to set them as constant? 
# yes like this:
#parameters_dict.update({
#    'epochs': {'value': 1}  
#    })

default_values = {
    "NUM_EPOCHS": 2000, # 2000 == 2e3
    #"NUM_EPOCHS": 2e5, # 2e5 == 200,000

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



# 1: Define objective/training function
def objective(config):
    print("in obj");
    print(f"{config=}")
    #print(f"{config.NN_SHAPE=}")
    print(f"\n{config.DECAY=}\n")

    print("\n\nCall a function to run the model and return the score\n\n\n")

    #score = config.x**3 + config.y
    score = 1

    return score


def main():
    print("enter main")
    #wandb.init(project="first-test-tubeballgame")
    wandb.init(project="trial-learn")
    score = objective(wandb.config)

#    wandb.log({"score": score})


# 2: Define the search space
sweep_configuration = {
    "method": "random",

    # metric is not needed for random search but here for now anyway
    "metric": {"goal": "minimize", "name": "score"},

    "parameters": {
        #"NUM_EPOCHS": 2000, # 2000 == 2e3
        #"NUM_EPOCHS": 2e5, # 2e5 == 200,000

        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},

        #"DECAY": 0.8,
        "DECAY": {"values": [0.8, 0.9, 0.95, 0.99]},
        #"LEARNING_RATE": 1e-3, # 1e-3 == 0.001
        #"BATCH_SIZE": 20,   

        "NN_SHAPE" : {"values":[
            ["I",  "I",  "I",  "I" ,"O"],
            ["I",  "I", "2I",  "I" ,"O"],
            ["I", "2I", "2I", "2I" ,"O"],
            ["I", "3I", "3I", "3I" ,"O"],
            ["I",  "I", "3I",  "I" ,"O"],
            ["I",  "I", "2I", "2I" ,"O"]
        ]},  

    },
}


#        "NN_SHAPE" : {"values":[
#            ["I",  "I",  "I",  "I" ,"O"],
#            ["I",  "I", "2I",  "I" ,"O"],
#            ["I", "2I", "2I", "2I" ,"O"],
#            ["I", "3I", "3I", "3I" ,"O"],
#            ["I",  "I", "3I",  "I" ,"O"],
#            ["I",  "I", "2I", "2I" ,"O"]
#        ]},   



print(f"{sweep_configuration=}")
for key in default_values:
    #if key not in sweep_configuration['parameters']:
    #    sweep_configuration['parameters'][key] = {'value': default_values[key]}
    # this next line does same as above 2 lines. Just practicing my pythonic wanking
    sweep_configuration['parameters'].setdefault(key, {'value': default_values[key]})

# set fixed values like this:
#parameters_dict.update({
#    'epochs': {'value': 1}  
#    })

##sweep_configuration['parameters'] = sweep_configuration['parameters'] | default_values
print(f"{sweep_configuration=}")


# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="trial-learn")
print(f"{sweep_id=}")

wandb.agent(sweep_id, function=main, count=2)
