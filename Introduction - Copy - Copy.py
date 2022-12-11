
def module(a):
    if a<0: return -a
    else:return a

def infinitNorm_for_vector(l):
    max = module(l[0])
    for i in l:
        if module(i) > max:
            max = module(i)
    return max
def Norm1(l):
    s = 0
    for i in l:
        s+=module(i)
    return s
def Norm2(l):
    jami = 0
    for i in l:
        jami+=module(i)**2

    jami = jami**(0.5)
    return jami
def Norm_of_frobenious(A):
    s = 0
    for i in A:
        s += (Norm2(i)**2)
    return s

def infinite_Norm_for_matrix(A):
    def oneNorm(l):
        s = 0
        for i in l:
            s+=module(i)
        return s
    max = oneNorm(A[0])
    for i in range(1,len(A)):
        if oneNorm(A[i])>max:
            max = oneNorm(A[i])
    return max
