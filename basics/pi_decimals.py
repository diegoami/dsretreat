
pi = 2
for i in range(1,100000):
    j = 4 * i ** 2
    pi = pi * (j) / (j - 1)
print (pi)
print("{:06.50f}".format(pi))


from functools import reduce
product = reduce((lambda pi, i: pi * (4*i**2)/(4*i**2 - 1)), range(1,100000), 2)
print(product)


l  = [(4*i**2)/(4*i**2 -1) for i in range(1,100000)]

product1 = reduce(lambda x, y: x * y, l, 2)
print(product1)
