import csv, random

with open('data/training_data.csv', 'w', newline='') as f:
  training_data_writer = csv.writer(f)
  for i in range(int(10e5)):
    x = random.random() * 5
    y = random.random() * 5
    fuzzy = random.random()
    c = int(fuzzy + x + y)
    training_data_writer.writerow([c, x, y])

with open('data/test_data.csv', 'w', newline='') as f:
  test_data_writer = csv.writer(f)
  for i in range(int(10)):
    x = random.random() * 5
    y = random.random() * 5
    fuzzy = random.random()
    c = int(fuzzy + x + y)
    test_data_writer.writerow([c, x, y])