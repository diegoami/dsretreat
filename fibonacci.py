def fib(n):
    if n < 0:
        raise ValueError
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

print(fib(30),fib(7))
print([fib(i) for i in range(20)])
#print(fib(-1))
