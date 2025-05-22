import math

def binom(n, m):
    return math.factorial(n) // (math.factorial(n - m) * math.factorial(m))

result = binom(2, 2) * binom(3, 2) * binom(4, 2) * binom(5, 2) * 7 ** 8

print(result // 1e9)