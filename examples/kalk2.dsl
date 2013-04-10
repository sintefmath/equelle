# Array literals, constant indexing, a for loop.
a = [ 2, 4, 7, 1 ]
b = a[0]  # == 2
c = [ 4, b ]
print c[1] # prints 2
d = [elem*elem for elem in a] # d = [ 4, 16, 49, 1]
print d[2] # prints 49
