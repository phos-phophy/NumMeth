import numpy as np


def forward_pass(matrix: np.ndarray,
                 f: np.ndarray,
                 select: bool = False) -> dict:
    correct = True
    count_perm = 0
    diag_elements = list()

    n = matrix.shape[0]
    perm = np.arange(n)

    inv_blank = np.diag(np.ones(n, dtype=np.float64))
    matrix_t = matrix.T

    for ind in range(n):
        tmp = matrix[ind][ind]

        if select or tmp == 0:
            i = np.argmax(np.abs(matrix[ind]))
            if i != ind:
                matrix_t[[i, ind]] = matrix_t[[ind, i]]

                count_perm += 1
                perm[[ind, i]] = i, ind

        diag_element = matrix[ind][ind]
        diag_elements.append(diag_element)

        if diag_element == 0:
            correct = False
            break

        matrix[ind] = matrix[ind] / diag_element
        inv_blank[ind] = inv_blank[ind] / diag_element
        f[ind] /= diag_element

        f[ind + 1:] -= matrix_t[ind][ind + 1:] * f[ind]
        inv_blank[ind + 1:] -= np.array([matrix_t[ind][ind + 1:]]).T \
                               @ np.array([inv_blank[ind]])
        matrix[ind + 1:] -= np.array([matrix_t[ind][ind + 1:]]).T \
                            @ np.array([matrix[ind]])

    det = (-1) ** count_perm
    for element in diag_elements:
        det *= element

    return {"correct": correct,
            "matrix": matrix,
            "f": f,
            "perm": perm,
            "det": det,
            "inv_blank": inv_blank}


def backward_pass(matrix: np.ndarray,
                  f: np.ndarray,
                  inv_blank: np.ndarray,
                  perm: np.ndarray) -> dict:
    n = matrix.shape[0]
    for ind in range(1, n):
        f[:n - ind] -= matrix.T[n - ind][:n - ind] * f[n - ind]
        inv_blank[:n - ind] -= np.array([matrix.T[n - ind][:n - ind]]).T \
                               @ np.array([inv_blank[n - ind]])

    return {"ans": f[perm],
            "inv_matrix": inv_blank}


def dif(first: np.ndarray, second: np.ndarray) -> float:
    tmp = first - second
    return np.sqrt(np.sum(tmp * tmp))


def get_equation() -> dict:
    ans = {"matrix": None,
           "f": None,
           "select": None}

    ch = input("Select the leading elements? y/n ")
    if ch == 'y':
        ans["select"] = True
    elif ch == 'n':
        ans["select"] = False
    else:
        print("Wrong input")
        return ans

    ch = input("\nWould you like to enter formulas for matrices "
               "or input file?\n"
               "Print 1 in the first case and 2 in the second: ")
    if ch == '1':
        try:
            n = int(input("\nFirst specify the count of "
                          "unknown variables: "))
            m = int(input("Now enter the value of m: "))

            ans["f"] = np.array([m * n - x ** 3 for x in range(1, n + 1)],
                                dtype=np.float64)
            ans["matrix"] = np.array([[(i + j) / (m + n) if i == j
                                       else n + m * m + j / m + i / n
                                       for j in range(1, n + 1)]
                                      for i in range(1, n + 1)],
                                     dtype=np.float64)

        except (ValueError, ZeroDivisionError):
            print("Wrong input")
            return ans

    elif ch == '2':
        try:

            n = int(input("\nFirst specify the count "
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
            ans["f"] = np.array([float(x) for x in f], dtype=np.float64)

        except ValueError:
            print("Wrong input")
            return ans

    else:
        print("Wrong input")

    return ans


def main():
    res = get_equation()

    matrix = res["matrix"]
    f = res["f"]
    select = res["select"]

    if matrix is None:
        return

    f_copy = f.copy()
    matrix_copy = matrix.copy()
    res = forward_pass(matrix, f, select)

    if res["correct"] is False:
        print("\nDeterminant is 0!")
        print("Violated the conditions of the problem")
        return

    perm = res["perm"]
    det = res["det"]

    matrix = res["matrix"]
    inv_blank = res["inv_blank"]
    f = res["f"]

    print("\nDeterminant is " + str(det))

    ans = backward_pass(matrix, f, inv_blank, perm)
    print("\nAnswer is ", end="")
    print(ans["ans"])
    print(f"\nResidual is {dif(matrix_copy @ ans['ans'].T, f_copy)}")

    cond_num = np.linalg.norm(matrix_copy, 2) * \
               np.linalg.norm(ans['inv_matrix'], 2)

    print(f"\nCondition number is {cond_num}")
    print("\nInverse matrix is")
    print(ans["inv_matrix"])


if __name__ == '__main__':
    main()
