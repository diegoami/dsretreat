
pi = 2
for i in range(1,100000):
    pi = pi * (4*i**2)/(4*i**2 - 1)
print (pi)


from functools import reduce
product = reduce((lambda pi, i: pi * (4*i**2)/(4*i**2 - 1)), range(1,100000), 2)
print(product)
