import numpy as np

def main():

    x = [5,150,150,5]
    y = [5,5,150,150]
    xp = [100,200,220,100]
    yp = [100,80,80,200]

    A = np.array([
        [-x[0],-y[0],-1,0,0,0,x[0]*xp[0],y[0]*xp[0],xp[0]],
        [0,0,0,-x[0],-y[0],-1,x[0]*yp[0],y[0]*yp[0],yp[0]],
        [-x[1],-y[1],-1,0,0,0,x[1]*xp[1],y[1]*xp[1],xp[1]],
        [0,0,0,-x[1],-y[1],-1,x[1]*yp[1],y[1]*yp[1],yp[1]],
        [-x[2],-y[2],-1,0,0,0,x[2]*xp[2],y[2]*xp[2],xp[2]],
        [0,0,0,-x[2],-y[2],-1,x[2]*yp[2],y[2]*yp[2],yp[2]],
        [-x[3],-y[3],-1,0,0,0,x[3]*xp[3],y[3]*xp[3],xp[3]],
        [0,0,0,-x[3],-y[3],-1,x[3]*yp[3],y[3]*yp[3],yp[3]],
    ], dtype=np.float64)

    U, S, V = singular_value_decomposition(A)

    print("U =\n", U)
    print("S =\n", S)
    print("V =\n", V)
    print("A == U*S*V' is", np.allclose(A,U.dot(S).dot(V.T)))
    print("Ax == 0 is", np.allclose(A.dot(V[:,-1]),np.zeros((9,1))))
    print("Homography matrix H =\n", V[:,-1].reshape((3,3)))

def singular_value_decomposition(A):

    m, n = A.shape
    r = min(m, n)
    
    ATA = A.T.dot(A)
    sortedEigVals, sortedEigVecs = sortedEig(ATA)
    zeroSingValsIndices = [i for i in range(r) if sortedEigVals[i] == 0]
    
    V = sortedEigVecs
    S = np.zeros((m,n))
    S[:r,:r] = np.diag(np.sqrt(sortedEigVals[:r]))
    U = np.empty((m,m))

    if m == n or m < n:
        for i in range(0, m - len(zeroSingValsIndices)):
            U[:,i] = A.dot(V[:,i]) / S[i,i]

        if zeroSingValsIndices:
            AAT = A.dot(A.T)
            _, sortedEigVecs2 = sortedEig(AAT)
            for i in zeroSingValsIndices:
                U[:,i] = sortedEigVecs2[:,i]
    elif m > n:
        for i in range(0, n - len(zeroSingValsIndices)):
            U[:,i] = A.dot(V[:,i]) / S[i,i]

        AAT = A.dot(A.T)
        _, sortedEigVecs2 = sortedEig(AAT)
        start = zeroSingValsIndices[0] if zeroSingValsIndices else n
        for i in range(start, m):
            U[:,i] = sortedEigVecs2[:,i]

    return (U, S, V)

def sortedEig(A):

    eigVals, eigVecs = np.linalg.eig(A)
    eigIndex = {eigVal: i for i, eigVal in enumerate(eigVals)}
    # sort eigen values in descending order
    eigVals[::-1].sort() 
    # sort corresponding eigen vectors
    tmp = eigVecs.copy()
    for i,val in enumerate(eigVals):
        eigVecs[:,i] = tmp[:,eigIndex[val]]

    return (eigVals, eigVecs)

if __name__ == '__main__':

    main()