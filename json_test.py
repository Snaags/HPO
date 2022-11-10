import json
import timeit
def load():
    with open("exp_temp.json","r") as f:
        x = json.load(f)

print(timeit.timeit(
                     stmt = load,
                     number = 100000))



