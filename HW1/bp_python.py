import math


x1, x2 = 0.26, 0.33
t1, t2 = 1, 0
lr = 0.5

w1, w2 = 0.1, 0.5
w3, w4 = 0.2, 0.4
w5, w6 = 0.4, 0.1

w7,  w8,  w9  = 0.1, 0.5, 0.3
w10, w11, w12 = 0.2, 0.1, 0.4

b1, b2, b3 = 0.3, 0.3, 0.3
b4, b5     = 0.7, 0.7


e = math.e
Relu = lambda x: x if x>0 else 0
dRelu = lambda x: 1 if x>0 else 0
Softmax = lambda x1, x2: (e**x1 / (e**x1 + e**x2), e**x2 / (e**x1 + e**x2)) 
CrossEntropy = lambda x1, x2, y1, y2: -y1*math.log(x1) - y2*math.log(x2)


def GD(p, grad, lr=0.5):
    return p - lr*grad


if __name__ == '__main__':
    for i in range(1):

        ## Forward
        h1 = w1*x1 + w2*x2 + b1
        h2 = w3*x1 + w4*x2 + b2
        h3 = w5*x1 + w6*x2 + b3

        a1 = Relu(h1)
        a2 = Relu(h2)
        a3 = Relu(h3)

        o1 = w7*a1  + w8*a2  + w9*a3  + b4
        o2 = w10*a1 + w11*a2 + w12*a3 + b5

        y1, y2 = Softmax(o1, o2)
        loss = CrossEntropy(y1, y2, t1, t2)


        ## Backward
        dL_o1 = (-t1/y1) * (y1*(1-y1)) + (-t2/y2) * (-y1*y2)
        dL_o2 = (-t1/y1) * (-y1*y2) + (-t2/y2) * (y2*(1-y2))

        dL_w7 = dL_o1 * a1
        dL_w8 = dL_o1 * a2
        dL_w9 = dL_o1 * a3
        dL_b4 = dL_o1 * 1

        dL_w10 = dL_o2 * a1
        dL_w11 = dL_o2 * a2
        dL_w12 = dL_o2 * a3
        dL_b5  = dL_o2 * 1

        dL_w1 = (dL_o1 * w7 + dL_o2 * w10) * dRelu(h1) * x1
        dL_w2 = (dL_o1 * w7 + dL_o2 * w10) * dRelu(h1) * x2
        dL_b1 = (dL_o1 * w7 + dL_o2 * w10) * dRelu(h1) * 1

        dL_w3 = (dL_o1 * w8 + dL_o2 * w11) * dRelu(h2) * x1
        dL_w4 = (dL_o1 * w8 + dL_o2 * w11) * dRelu(h2) * x2
        dL_b2 = (dL_o1 * w8 + dL_o2 * w11) * dRelu(h2) * 1

        dL_w5 = (dL_o1 * w9 + dL_o2 * w12) * dRelu(h3) * x1
        dL_w6 = (dL_o1 * w9 + dL_o2 * w12) * dRelu(h3) * x2
        dL_b3 = (dL_o1 * w9 + dL_o2 * w12) * dRelu(h3) * 1

        w1  = GD(w1, dL_w1)
        w2  = GD(w2, dL_w2)
        w3  = GD(w3, dL_w3)
        w4  = GD(w4, dL_w4)
        w5  = GD(w5, dL_w5)
        w6  = GD(w6, dL_w6)
        w7  = GD(w7, dL_w7)
        w8  = GD(w8, dL_w8)
        w9  = GD(w9, dL_w9)
        w10 = GD(w10, dL_w10)
        w11 = GD(w11, dL_w11)
        w12 = GD(w12, dL_w12)
        b1  = GD(b1, dL_b1)
        b2  = GD(b2, dL_b2)
        b3  = GD(b3, dL_b3)
        b4  = GD(b4, dL_b4)
        b5  = GD(b5, dL_b5)


    print('out1:', round(y1, 2))
    print('out2:', round(y2, 2))
    print('loss:', round(loss, 2))
    print('w1:', round(w1, 2))
    print('w2:', round(w2, 2))
    print('w3:', round(w3, 2))
    print('w4:', round(w4, 2))
    print('w5:', round(w5, 2))
    print('w6:', round(w6, 2))
    print('w7:', round(w7, 2))
    print('w8:', round(w8, 2))
    print('w9:', round(w9, 2))
    print('w10:', round(w10, 2))
    print('w11:', round(w11, 2))
    print('w12:', round(w12, 2))
    print('b1:', round(b1, 2))
    print('b2:', round(b2, 2))
    print('b3:', round(b3, 2))
    print('b4:', round(b4, 2))
    print('b5:', round(b5, 2))

    print('dL_w1:', round(dL_w1, 2))
    print('dL_w2:', round(dL_w2, 2))
    print('dL_w3:', round(dL_w3, 2))
    print('dL_w4:', round(dL_w4, 2))
    print('dL_w5:', round(dL_w5, 2))
    print('dL_w6:', round(dL_w6, 2))
    print('dL_w7:', round(dL_w7, 2))
    print('dL_w8:', round(dL_w8, 2))
    print('dL_w9:', round(dL_w9, 2))
    print('dL_w10:', round(dL_w10, 2))
    print('dL_w11:', round(dL_w11, 2))
    print('dL_w12:', round(dL_w12, 2))
    print('dL_b1:', round(dL_b1, 2))
    print('dL_b2:', round(dL_b2, 2))
    print('dL_b3:', round(dL_b3, 2))
    print('dL_b4:', round(dL_b4, 2))
    print('dL_b5:', round(dL_b5, 2))
