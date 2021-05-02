# https://matplotlib.org/
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 100, 15
data_set = mu + sigma * np.random.randn(1000)

# the histogram of the data
n, bins, patches = plt.hist(data_set, 50, facecolor='g', alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title("Histogram of IQ")
plt.text(50, .025, r'$\mu=100, \ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
# plt.show()
print("***********")

print("*** Figures and Subplots ***")
my_first_figure = plt.figure("My First Figure")
subplot_1 = my_first_figure.add_subplot(2, 3, 1)
subplot_2 = my_first_figure.add_subplot(2, 3, 6)
# plt.plot(np.random.rand(50).cumsum(), 'k--')
# plt.show()

subplot_2 = my_first_figure.add_subplot(2, 3, 2)
plt.plot(np.random.rand(50), 'go')
# plt.show()

print("*** Multiples Lines, Single Plot ***")
data_set_size = 15
low_mu, low_sigma = 50, 4.3
low_data_set = low_mu + low_sigma * np.random.randn(data_set_size)
high_mu, high_sigma = 57, 5.2
high_data_set = high_mu + high_sigma * np.random.randn(data_set_size)

days = list(range(1, data_set_size + 1))
plt.plot(days, low_data_set, days, high_data_set)
plt.show()