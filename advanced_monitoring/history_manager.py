# advanced_monitoring/history_manager.py

sensor_history = []

def add_reading(reading):
    sensor_history.append(reading)

    # Keep last 10 readings only
    if len(sensor_history) > 10:
        sensor_history.pop(0)

def get_last_readings(n=3):
    return sensor_history[-n:]