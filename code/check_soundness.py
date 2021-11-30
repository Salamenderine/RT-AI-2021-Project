import numpy as np

def compute_linear_bounds(lower, upper):
    w_l, w_u, b_l, b_u = 0, 0, 0, 0
    if lower >=0 and upper >= 0:
        w_l = upper + lower
        b_l = - 0.25 * (upper + lower)**2 - 0.5
        w_u = upper + lower
        b_u = - upper * lower - 0.5

    elif lower * upper < 0:
        if -lower <= upper:
            exp_l = np.exp(lower)
            w_l = upper + lower
            b_l = - 0.25 * (upper + lower)**2 - 0.5
            w_u = (upper**2 - 0.5 + exp_l/(1 + exp_l))/(upper - lower)
            b_u = - w_u * lower - exp_l/(1 + exp_l)
        else:
            exp_l = np.exp(lower)
            w_l = - (-0.5 + exp_l/(1 + exp_l))/lower
            b_l = - 0.5
            u1 = - exp_l/((1 + exp_l)**2)
            u2 = (upper**2 - 0.5 + exp_l/(1 + exp_l))/(upper - lower)
            w_u = max(u1, u2)
            b_u = - w_u * lower - exp_l/(1 + exp_l)
    else:
        mid = 0.5 * (upper + lower)
        exp_mid = np.exp(mid)
        exp_u = np.exp(upper)
        exp_l = np.exp(lower)
        w_l = ((- exp_u/(1 + exp_u)) + (exp_l/(1 + exp_l)))/(upper - lower)
        b_l = - w_l * lower - (exp_l/(1 + exp_l))
        w_u = (-exp_mid)/((1 + exp_mid)**2)
        b_u = - w_u * mid - (exp_mid)/(1 + exp_mid) 
    return (w_l, b_l), (w_u, b_u)


def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig


def SPU(x):
    if x >= 0:
        res = x**2 - 0.5
    else:
        res = sigmoid(-x) -1
    return res


if __name__ == "__main__":
    for _ in range(10000):
        l, u = np.random.randn(), np.random.randn()
        if l > u:
            l, u = u, l
        
        (w_l, b_l), (w_u, b_u) = compute_linear_bounds(l, u)

        # for x in np.linspace(l, u, 10000):
        for x in np.linspace(l, u, 1000):
            assert SPU(x) > w_l * x + b_l - 1e-5
            # if SPU(x) > w_l * x + b_l - 1e-5 == False:
            #     print("the point is:", x)
            #     print("l and u are:", l, u)
            #     break
        for x in np.linspace(l, u, 10000):
            assert SPU(x) < w_u * x + b_u + 1e-5
            # if SPU(x) < w_u * x + b_u + 1e-5 == False:
            #     print("the point is:", x)
            #     print("l and u are:", l, u)
            #     break

