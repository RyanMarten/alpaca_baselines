import json
import random

# Set the random seed
random.seed(314)

# Load the original JSON file
with open('alpaca_data.json', 'r') as f:
    data = json.load(f)

# Check if the data is a list and has at least 10,000 objects
if not isinstance(data, list):
    raise ValueError("The JSON file should contain a list of objects")

if len(data) < 10000:
    raise ValueError(f"The JSON file contains only {len(data)} objects, which is less than 10,000")

# Take a random subset of 10,000 objects
subset = random.sample(data, 10000)

# Save the subset to a new JSON file
with open('alpaca_data_10k.json', 'w') as f:
    json.dump(subset, f, indent=2)

print(f"Successfully created alpaca_data_10k.json with 10,000 randomly selected objects using seed 314")
