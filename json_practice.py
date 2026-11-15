import json


with open("hyperparameter_grid.json", "r") as file:
    data = json.load(file)

print(data["learning_rate"][0])

data = {
    "name": "sathiyajith",
    "rollno": 56,
    "cgpa": 8.6,
    "phone": "9976770500"
}

json_str = json.dumps(data, indent=4)
with open("hyperparameter_grid.json", "w") as f:
    f.write(json_str)