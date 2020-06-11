import numpy as np 

def optimal_rule(inputs, label, num_class, step_sz): 

    n = len(inputs)
    m = len(inputs[0])

    X = []
    for i in range(num_class): X.append([])
    for i in range(n): 
        X[label[i]].append(inputs[i])
   

    # ind[fired][label]
    ind = np.zeros((2, num_class, m))
    for i in range(n): 
        ind[1, label[i]*1] += (np.sign(inputs[i]) == 1) * step_sz
        ind[0, label[i]*1] += (np.sign(inputs[i]) == 0) * step_sz

    for x in [0,1]: 
      for y in [0,1]: 
        print(x,y, ind[x, y])

    # partial derivate of loss w.r.t.  r(a,b)
    # return coefficients of r(0,0), r(0,1), r(1,0), r(1,1)
    def grad(a, b): 
        r = np.array([[0,0],[0,0]])
        const = 0 

        for i in range(num_class):
            for xj in X[i]: 

                for k in range(num_class): 
                    c = np.dot(xj, grad_w(k, a, b))
                    if k == i: const -= c 

                    coefs = W_coef(k)    
                    for x in [0,1]: 
                        for y in [0,1]: 
                            r[x][y] += (np.dot(coefs[x][y], xj)) * c
        return r, const
                        
    # Return partial derivative of W_l w.r.t. r(a,b)
    def grad_w(l, a, b): 
        if b == 0: 
            return ind[a][l]
        else: 
            c = 0
            for i in range(num_class): 
                if i == l: continue 
                c += ind[a][i]
            return c          

    def W_coef(l): 
        r = [[0,0],[0,0]]

        r[1][0] = ind[1][l]
        r[0][0] = ind[0][l]

        for i in range(num_class): 
            if i == l: continue
            r[1][1] += ind[1][i]
            r[0][1] += ind[0][i]

        return r 

    def eval_loss(r): 
        loss = 0
        for i in range(num_class):
            for xj in X[i]: 
                for k in range(num_class): 
                    coefs = np.array(W_coef(k))

                    c = 0
                    for x in [0,1]: 
                        for y in [0,1]:
                            c += r[2 * x + y] * np.dot(coefs[x,y], xj)

                    if i == k: 
                        loss += (c - 1) ** 2
                    else: 
                        loss += c ** 2
        return loss / (n * num_class)

    A = np.zeros((4,4))
    b = np.zeros(4) 
    i = 0
    for x in [0,1]: 
        for y in [0,1]: 
            a1, b[i] = grad(x,y)
            b[i] *= -1 

            A[i] = a1.flatten()
            i += 1
    
    A /= n 
    b /= n
    sol = np.linalg.solve(A, b)

    print("LOSS:", eval_loss(sol))

    return sol 