import numpy as np


# y = wx + b


def cal_err_linear_given_points(b, w, points):
    totalErr = 0
    for i in range (0, len (points)):
        x = points[i, 0]
        y = points[i, 1]
        totalErr += (y - (w * x + b)) ** 2
    return totalErr / float (len (points))


def step_grad(b_current, w_current, points, learnRate):
    b_grad = 0
    w_grad = 0
    N = float (len (points))

    for i in range (0, len (points)):
        x = points[i, 0]
        y = points[1, 1]
        b_grad += -(2 / N) * (y - ((w_current * x) + b_current))
        w_grad += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learnRate * b_grad)
    new_w = w_current - (learnRate * w_grad)
    return [new_b, new_w]


# iterate to optimize
def grad_descent_exe(points, starting_b, starting_w, learnRate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range (num_iterations):
        b, w = step_grad (b, w, np.array (points), learnRate)
        print (b, w)
    return [b, w]


def exe():
    points = np.genfromtxt ("data.csv", delimiter=",")
    # print(points)
    learnRate = 0.0001
    initial_b = 0  # guessing y-intercept
    initial_w = 0  # guessing slope
    num_iterations = 1000
    print ("Starting grad descent at b = {0}, m = {1}, err = {2}"
           .format (initial_b, initial_w, cal_err_linear_given_points (initial_b, initial_w, points)))
    [b, w] = grad_descent_exe (points, initial_b, initial_w, learnRate, num_iterations)
    print ("After {0} interations b = {1}, m = {2}, err = {3}"
           .format (num_iterations, b, w, cal_err_linear_given_points (b, w, points)))


if __name__ == '__main__':
    exe ()
