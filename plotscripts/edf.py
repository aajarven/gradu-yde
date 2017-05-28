import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(715517)

N = 150
normalNumbers = np.random.normal(10.0, 1.2, N)
normalNumbersLow = np.random.normal(5.0, 1.2, N)
normalNumbersWide = np.random.normal(10.0, 4.0, N)

percentage = np.arange(0, 100, 100.0/N)
percentage = np.append(percentage, 100)

normalNumbers.sort()
normalNumbersLow.sort()
normalNumbersWide.sort()

biggest = max(max(np.amax(normalNumbers), np.amax(normalNumbersLow)),
        np.amax(normalNumbersWide))
normalNumbers = np.append(normalNumbers, biggest)
normalNumbersLow = np.append(normalNumbersLow, biggest)
normalNumbersWide = np.append(normalNumbersWide, biggest)

plt.plot(normalNumbers, percentage)
plt.plot(normalNumbersLow, percentage)
plt.plot(normalNumbersWide, percentage)

plt.xlabel('values')
plt.ylabel('CDF')

plt.show();
