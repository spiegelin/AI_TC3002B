def activation_function(x, prev_u):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return prev_u

def recall(test_pattern, T):
    n = len(test_pattern)
    U = test_pattern[:]
    
    while True:
        U_next = [0] * n
        
        for j in range(n):
            dot_product = 0
            for i in range(n):
                dot_product += U[i] * T[i][j]
            U_next[j] = activation_function(dot_product, U[j])
            
        if U == U_next:
            return U_next
            
        U = U_next[:]

def main():
    patterns = [
        [1, 1, 1, -1],
        [-1, -1, -1, 1]
    ]

    n = len(patterns[0])
    T = [[0 for _ in range(n)] for _ in range(n)]

    for p in patterns:
        for i in range(n):
            for j in range(n):
                T[i][j] += p[i] * p[j]

    for i in range(n):
        T[i][i] = 0

    print("Weight Matrix T:")
    for row in T:
        print(row)
    print()

    tests = [
        [1, 1, 1, -1],
        [-1, -1, -1, -1]
    ]

    for test in tests:
        result = recall(test, T)
        print(f"Input: {test} -> {result}")

if __name__ == "__main__":
    main()
