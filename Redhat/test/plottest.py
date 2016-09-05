import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot([0,0,1,1],[-10, 20,-10, 10])
plt.subplot()
plt.ylabel('some numbers')
plt.title("hello")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
x = [0, 1, 2, 4, 5, 6]
y = [1, 2, 3, 2, 4, 1]
plt.plot(x, y, '-*r')
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

x = [0, 1, 2, 4, 5, 6]
y = [1, 2, 3, 2, 4, 1]
z = [1, 2, 3, 4, 5, 6]
plt.plot(x, y, '--*r', x, z, '-.+g')
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("hello world")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
x = [0, 1, 2, 4, 5, 6]
y = [1, 2, 3, 2, 4, 1]
z = [1, 2, 3, 4, 5, 6]
plt.bar(x, y)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()