from advanced_monitoring.history_manager import add_reading
from advanced_monitoring.advanced_engine import advanced_monitor

# Simulate 3 low readings
add_reading({"N": 30, "P": 20, "K": 20})
add_reading({"N": 28, "P": 22, "K": 19})
add_reading({"N": 29, "P": 18, "K": 21})

result = advanced_monitor("rice")

print(result)