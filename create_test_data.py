import csv
import numpy as np
import random

with open('data/training_data.csv', 'w', newline='') as f:
  training_data_writer = csv.writer(f)
  for i in range(int(10e5)):
    x = random.random() * 5
    y = random.random() * 5
    fuzzy = np.random.normal(1, 0.5)
    fuzzy = 0 if fuzzy < 0 else fuzzy
    c = int(fuzzy + x + y)
    training_data_writer.writerow([c, x, y])

with open('data/test_data.csv', 'w', newline='') as f:
  test_data_writer = csv.writer(f)
  for i in range(int(1000)):
    x = random.random() * 5
    y = random.random() * 5
    fuzzy = np.random.normal(1, 0.5)
    fuzzy = 0 if fuzzy < 0 else fuzzy
    c = int(fuzzy + x + y)
    test_data_writer.writerow([c, x, y])
