def f0(x: float, y: list) -> float:
    return -y[0] - x**2


def f1(x: float, y: list) -> float:
    return 3 - y[0] - x


def f2(x: float, y: list) -> float:
    return y[0] - y[0] * x


def f3(x: float, y: list) -> float:
    return -2 * x * y[0] ** 2 + y[1] ** 2 - x - 1


def f4(x: float, y: list) -> float:
    return 1 / (y[1] ** 2) - y[0] - x / y[0]


def runge_kutta2(x: float, y: list, count_y: int,
                 f: list, h: float) -> list:
    y_ = [y[i] + f[i](x, y) * h for i in range(count_y)]
    return [y[i] + (f[i](x, y) + f[i](x + h, y_)) / 2 * h
            for i in range(count_y)]


def runge_kutta4(x: float, y: list, count_y: int,
                 f: list, h: float) -> list:
    k1 = [f[i](x, y) for i in range(count_y)]

    tmp_y = [y[i] + h / 2 * k1[i] for i in range(count_y)]
    k2 = [f[i](x + h / 2, tmp_y) for i in range(count_y)]

    tmp_y = [y[i] + h / 2 * k2[i] for i in range(count_y)]
    k3 = [f[i](x + h / 2, tmp_y) for i in range(count_y)]

    tmp_y = [y[i] + h * k3[i] for i in range(count_y)]
    k4 = [f[i](x + h, tmp_y) for i in range(count_y)]
    return [y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            for i in range(count_y)]


def main():
    try:
        print("Test 1:\ny' = -y + x^2\ny(0) = 10")
        print("\nTest 2:\ny' = 3 - y - x\ny(0) = 0")
        print("\nTest 3:\ny' = y - yx\ny(0) = 5")
        print("\nTest 4:\nu' = -2x(u^2) + v^2 - x - 1\n"
              "v' = 1/(v^2) - u - x/u\nu(0) = 1\nv(0) = 1\n")
        test = int(input("Choose the test (1, 2, 3, or 4): "))
        print()

        steps_number = int(input("Enter the steps number: "))
        step_size = float(input("Enter the size of step: "))
        print()

        if test not in [1, 2, 3, 4]:
            raise ValueError

    except ValueError:
        print("Wrong input")
        return

    if test == 1:
        x, f, count_y = 0, [f0], 1
        y2, y4 = [10], [10]
        print("x\t\ty (RK-2)\t\ty (RK-4)")
        print(f"{x:.3f}\t{y2[0]:.10f}\t{y4[0]:.10f}")
    elif test == 2:
        x, f, count_y = 0, [f1], 1
        y2, y4 = [0], [0]
        print("x\t\ty (RK-2)\t\ty (RK-4)")
        print(f"{x:.3f}\t{y2[0]:.10f}\t{y4[0]:.10f}")
    elif test == 3:
        x, f, count_y = 0, [f2], 1
        y2, y4 = [5], [5]
        print("x\t\ty (RK-2)\t\ty (RK-4)")
        print(f"{x:.3f}\t{y2[0]:.10f}\t{y4[0]:.10f}")
    else:
        x, f, count_y = 0, [f3, f4], 2
        y2, y4 = [1, 1], [1, 1]
        print("x\t\ty_1 (RK-2)\t\ty_2 (RK-2)"
              "\t\ty_1 (RK-4)\t\ty_2 (RK-4)")
        print(f"{x:.3f}\t{y2[0]:.10f}\t{y2[1]:.10f}"
              f"\t{y4[0]:.10f}\t{y4[1]:.10f}")

    for step in range(steps_number):
        y2 = runge_kutta2(x, y2, count_y, f, step_size)
        y4 = runge_kutta4(x, y4, count_y, f, step_size)
        x += step_size

        print(f"{x:.3f}\t", end="")
        for i in range(count_y):
            print(f"{y2[i]:.10f}\t", end="")
        for i in range(count_y):
            print(f"{y4[i]:.10f}\t", end="")
        print()


if __name__ == '__main__':
    main()
