import tensorflow as tf
import matplotlib.pyplot as plt

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self,warmup_steps = 20000):
    super(CustomSchedule, self).__init__()

    self.warmup_steps = warmup_steps
    self.LONG_SCHEDULE = {
        'step_values': [10000, 30000, 40000,  50000, 60000 ],
        'learning_rates': [0.001, 0.01,  0.01/2, (0.01/2)/2, ((0.01/2)/2)/2, (((0.01/2)/2)/2)/2],
        'momentum': 0.9,
        'momentum2': 0.999,
        'weight_decay': 0.0004,
        'max_iter':75000,
    }
  def __call__(self, step):
    arg = (step%self.warmup_steps)*(1 / (1 +  step)) * 0.01

    return arg

learning_rate = CustomSchedule()

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

temp_learning_rate_schedule = CustomSchedule()

plt.plot(temp_learning_rate_schedule(tf.range(70000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()

