"""
LONG_SCHEDULE = {
    'step_values': [10000, 20000, 30000, 40000, 50000],
    'learning_rates': [0.001, 0.01, 0.01/2, (0.01/2)/2, ((0.01/2)/2)/2, (((0.01/2)/2)/2)/2],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.00004,
    'max_iter': 75000,
}
"""

LONG_SCHEDULE = {
    'step_values': [40000, 80000, 120000, 16000, 200000],
    'learning_rates': [0.001, 0.01, 0.01/2, (0.01/2)/2, ((0.01/2)/2)/2, (((0.01/2)/2)/2)/2],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.00004,
    'max_iter': 300000,
}

FINETUNE_SCHEDULE = {
    # TODO: Finetune schedule
}