import math

test1 = {
    "description": "y'' - xy' + 2y = x - 1\n"
                   "y(0.9) - 0.5y'(0.9) = 2\ny(1.2) = 1\n",
    "p": lambda x: -x,
    "q": lambda x: 2,
    "f": lambda x: 1 - x,
    "a": 0.9, "b": 1.2,
    "s1": 1, "g1": -0.5, "d1": 2,
    "s2": 1, "g2": 0, "d2": 1
}

test2 = {
    "description": "y'' - 0.5y' - 3y = 2x^2\n"
                   "y(1) - 2y'(1) = 0.6\ny(1.3) = 1\n",
    "p": lambda x: -0.5,
    "q": lambda x: -3,
    "f": lambda x: -2 * x * x,
    "a": 1, "b": 1.3,
    "s1": 1, "g1": -2, "d1": 0.6,
    "s2": 1, "g2": 0, "d2": 1
}

test3 = {
    "description": "y'' + y' = 1\ny'(0) = 0\ny(1) = 1\n",
    "p": lambda x: 1,
    "q": lambda x: 0,
    "f": lambda x: -1,
    "a": 0, "b": 1,
    "s1": 0, "g1": 1, "d1": 0,
    "s2": 1, "g2": 0, "d2": 1
}

test4 = {
    "description": "y'' + y = 1\ny'(0) = 0\n"
                   "y(0.5 * pi) - y'(0.5 * pi) = 2\n",
    "p": lambda x: 0,
    "q": lambda x: 1,
    "f": lambda x: -1,
    "a": 0, "b": math.pi/2,
    "s1": 0, "g1": 1, "d1": 0,
    "s2": 1, "g2": -1, "d2": 2
}


def forward_pass(test: dict, n: int):
    h = (test["b"] - test["a"]) / n

    C = test["g1"] / h
    B = test["s1"] - C
    D = test["d1"]

    alpha = [-C/B]
    beta = [D/B]

    x = test["a"] + h
    for i in range(n - 1):
        A = 1 / h / h - test["p"](x) / 2 / h
        B = -2 / h / h + test["q"](x)
        C = 1 / h / h + test["p"](x) / 2 / h
        D = -test["f"](x)

        alpha.append(-C / (A * alpha[i] + B))
        beta.append((D - A * beta[i]) / (A * alpha[i] + B))
        x += h

    return alpha, beta


def backward_pass(test: dict, n: int, alpha: list, beta: list) -> list:
    h = (test["b"] - test["a"]) / n

    A = -test["g2"] / h
    B = test["s2"] - A
    D = test["d2"]

    y = [(D - A * beta[n-1]) / (A * alpha[n-1] + B)]
    for i in range(n):
        y.append(alpha[n - i - 1] * y[i] + beta[n - i - 1])

    y.reverse()
    return y


def main():
    tests_dict = [test1, test2, test3, test4]

    for ind, test in enumerate(tests_dict):
        print(f'Test {ind + 1}:')
        print(test["description"])

    test_num = int(input("\nChoose the test (1, 2, 3 or 4): "))
    if test_num > len(tests_dict):
        print("Wrong input")
        return

    test = tests_dict[test_num - 1]

    try:
        n = int(input("Enter the count of steps: "))
        if n <= 0:
            raise ValueError
    except ValueError:
        print("Wrong input")
        return

    alpha, beta = forward_pass(test, n)
    y = backward_pass(test, n, alpha, beta)

    h = (test["b"] - test["a"]) / n
    x = test["a"]

    print("x\t\t\ty")
    for i in y:
        print(f'{x:.5f}\t\t{i:.10f}')
        x += h


if __name__ == '__main__':
    main()
