import ast

def readdata():
    f = open("delayvscost.txt", "r")
    y = []
    err = []
    for d in f:
        y.append(float(d.split()[0]))
        err.append(float(d.split()[1]))
    return y, err
    #print(f.readline()
def checkfile_empty():
    f = open("estimation.txt", "r")
    if f.read() == '':
        print("yes")

def testdic():
    d = {[1,2,3]: 0, [2, 3,1]:1}
    return d[[1,2,3]]
def testlist():
    alist = [[] for i in range(10)]
    if not any(alist):
        print("Empty list")


if __name__ == '__main__':
    #print(readdata())
    #checkfile_empty()
    # f = open("estimation.txt", 'r')
    # #print(f.readline())
    # str = f.readline()
    # #print(list(str))
    #
    # #print(eval(f.readline()))
    # list = ast.literal_eval(str)
    # print(list[1][2])
    #
    # print(ast.literal_eval(str))
    # f.close()
    print([float(r) for r in open("avgrestime.txt", 'r').readlines()])
    testlist()