import random
def qsort(arr):
    less = []
    greater = []
    if (len(arr) < 2):
        return arr
    else:
        i = random.randint(1,len(arr))
        pvt = arr.pop(i-1)
        for x in arr:
            if x < (pvt+1):
                less.append(x)
            else:
                greater.append(x)
    return qsort(less)+[pvt]+qsort(greater)



print(qsort([37,20,89,38,81,94,17,64,54,86]))
print(qsort([21,45,86]))
print(qsort([1]))
print(qsort([1]))
print(qsort([13,40,89,56,44,81,40,9,97,81,28,31,3,5,18,79,70,63,39,87]))

        