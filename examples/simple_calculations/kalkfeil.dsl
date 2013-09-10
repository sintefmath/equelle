a = 1
b = 2
c = a / b / 4   # Error, ambiguous!
c = a / (b * 4) # Correct.
print c, d


a = 1
b = 2
a = b*3   # Error, redefinition!
print a
