import json

x = {"lr": 0.005, "batch_size": 32, "Date": "11-09-2022"}

with open("json_test_data.json","w") as f:
  json.dump(x,f)
