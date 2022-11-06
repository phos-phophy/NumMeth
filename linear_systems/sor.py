import numpy as np


def relax(prev_x: np.ndarray,
          n: int,
          matrix: np.ndarray,
          f: np.ndarray,
          w: float) -> np.ndarray:

    x = np.zeros(prev_x.shape, dtype=np.float64)
    for i in range(n):
        sum1 = sum(-x[j] * w * matrix[i][j] / matrix[i][i]
                   for j in range(i))
        sum2 = sum(-prev_x[j] * w * matrix[i][j] / matrix[i][i]
                   for j in range(i, n))
        x[i] = sum1 + sum2 + f[i] * w / matrix[i][i] + prev_x[i]
    return x


def get_equation() -> dict:
    ans = {"matrix": None, "f": None,
           "n": None, "res": None, "w": None}
    try:
        ans["res"] = float(input("First specify residual: "))
        ans["w"] = float(input("Enter the value of w: "))
        if ans["w"] <= 0 or ans["w"] >= 2:
            raise ValueError

        ch = input("\nWould you like to enter formulas for matrices "
                   "or input file?\n"
                   "Print 1 in the first case and 2 in the second: ")

        if ch == '1':
            n = int(input("\nNow specify the count of "
                          "unknown variables: "))
            m = int(input("Now enter the value of m: "))

            ans["f"] = np.array([m * n - x ** 3 for x in range(1, n + 1)],
                                dtype=np.float64)
            ans["matrix"] = np.array([[(i + j) / (m + n) if i == j
                                       else n + m * m + j / m + i / n
                                       for j in range(1, n + 1)]
                                      for i in range(1, n + 1)],
                                     dtype=np.float64)
            ans["n"] = n
        elif ch == '2':
            n = int(input("\nNow specify the count "
                          "of unknown variables: "))
            file = input("Enter file name: ")

            matrix = []
            f = []

            with open("tests/" + file, "r") as file:
                for line in file:
                    nums = [float(x) for x in line.strip().split()]
                    if len(nums) != n + 1:
                        print("Wrong input")
                        return ans

                    matrix.append(nums[:n])
                    f.append(nums[n])

            ans["matrix"] = np.array(matrix, dtype=np.float64)
            ans["f"] = np.array([f], dtype=np.float64).T
            ans["n"] = n
        else:
            print("Wrong input")

    except (ValueError, ZeroDivisionError):
        print("Wrong input")
        return ans

    return ans


def dif(first: np.ndarray, second: np.ndarray) -> float:
    tmp = first - second
    return np.sqrt(np.sum(tmp * tmp))


def main():
    MAX_T = 300
    D = 1e-6

    ans = get_equation()

    matrix = ans["matrix"]
    f = ans["f"]
    n = ans["n"]
    res = ans["res"]
    w = ans["w"]

    if matrix is None:
        return

    if np.linalg.det(matrix) == 0:
        print("\nDeterminant is 0!")
        print("Violated the conditions of the problem")
        return

    f = matrix.T @ f
    matrix = matrix.T @ matrix

    t = 0
    iterations = 0

    x = np.zeros((n, 1), dtype=np.float64)
    residual = dif(matrix @ x, f)

    while residual > res and t < MAX_T:
        iterations += 1
        prev_x = x
        x = relax(prev_x, n, matrix, f, w)
        residual = dif(matrix @ x, f)

        if dif(prev_x, x) < D:
            t += 1
        else:
            t = 0

    print(f'\n{iterations} iterations')
    print(f'\nResidual is {residual:.6f}')
    print("\nAnswer is")
    for x_i in x:
        print(x_i[0])


if __name__ == '__main__':
    main()
