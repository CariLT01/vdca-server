import json

# Step 1: Load JSON from a file
with open("text.txt", "r", encoding="utf-8") as f:
    data = json.load(f)

# Optional: modify data
# data["new_key"] = "new_value"

# Step 2: Export JSON to another file
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(data, f)

print("JSON loaded and exported successfully.")
