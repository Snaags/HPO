import numpy as np 
import torch 
import random 
import json


def set_seed(JSON_PATH):
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    if "SEED" in data:
            seed = data["SEED"]
    else:
            seed = random.randint(0,999)
            data["SEED"] = seed
            with open(JSON_PATH, "w") as f:
                json.dump(data, f)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
