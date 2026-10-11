from maintenance_engine import monitor_crop

sample_data = {
    "N": 50,
    "P": 30,
    "K": 25,
    "moisture": 0.35,
    "temperature": 35
}

result = monitor_crop("rice", sample_data)

print(result)