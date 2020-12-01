import requests
import time
import sys
import random
import subprocess
import re
import socket
import numpy as np
from math import *
import matplotlib.pyplot as plt
import random
import numpy
import scipy.stats
import ast
from itertools import permutations
from matplotlib.ticker import FuncFormatter
from ast import literal_eval

inf = 100000.0
rate = 0
indexCloud = 9
numServices = 3
numNodes = 6
TCP_PORT = 5005
MESSAGE = "p"
BUFFER_SIZE = 1024


def generateTraffic(i, rate, type, lam):

	start = time.time()
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.connect(("10.0.0.%i"%(i+1), TCP_PORT))
	sendtime = ' ' + str(time.time())
	for i in range(lam):
		s.send(MESSAGE)
		if type == "possion":
			time.sleep(numpy.random.poisson(rate))
		elif type == "normal30":
			time.sleep(numpy.random.normal(rate, 0.3*rate))
		elif type == "normal20":
			time.sleep(numpy.random.normal(rate, 0.2*rate))
		elif type == "normal10":
			time.sleep(numpy.random.normal(rate, 0.1*rate))
	s.send(sendtime)
	s.shutdown(socket.SHUT_WR)
	#s.send()
	# while True:
	# 	data = s.recv(BUFFER_SIZE)
	# 	if data:
	# 		print(data)
	# 		break
	# 	else:
	# 		print('no data received')
	#data = s.recv(BUFFER_SIZE)
	#print(data)
	s.close()
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.bind(("0.0.0.0", TCP_PORT))
	s.listen(1)
	conn, addr = s.accept()
	d = conn.recv(BUFFER_SIZE)
	ret = d + ' ' + str(time.time())
	#print(ret)
	print("Time stamps:",parseRet(ret))
	s.shutdown(socket.SHUT_WR)
	s.close()
	t = time.time()-start
	#print("complete, rtt:", t)
	return parseRet(ret), t

def parseRet(str):
	strlist = str.split()
	timelist = strlist[1:]
	delaylist = []
	for i, s in enumerate(timelist[:-1]):
		delaylist.append(float(timelist[i+1]) - float(timelist[i]))
	return delaylist

def sendingPackets(i, rate):
	start = time.time()
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect(("10.0.0.%i"%(i+1), TCP_PORT))
	for i in range(100):
		s.send(MESSAGE)
		time.sleep(rate)
	s.shutdown(socket.SHUT_WR)
	#s.send()
	# while True:
	# 	data = s.recv(BUFFER_SIZE)
	# 	if data:
	# 		print(data)
	# 		break
	# 	else:
	# 		print('no data received')
	#data = s.recv(BUFFER_SIZE)
	#print(data)
	s.close()
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(("0.0.0.0", TCP_PORT))
	s.listen(1)
	conn, addr = s.accept()
	d = conn.recv(BUFFER_SIZE)
	print(d)
	s.close()
	ete = time.time()-start
	#print("complete, rtt:", t)
	return ete

def sendEstimatePackets(i, rate):
	start = time.time()
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect(("10.0.0.%i" % (i + 1), TCP_PORT))
	for i in range(100):
		s.send(MESSAGE)
		time.sleep(rate)
	s.shutdown(socket.SHUT_WR)
	# while True:
	# 	data = s.recv(BUFFER_SIZE)
	# 	print("received data:", data)
	# 	if not data: break
	data = s.recv(BUFFER_SIZE)
	s.close()
	t = time.time()-start
	print(data)
	return t, float(data)

def trafficGenerationRate():
	lp = [4, 5, 6]
	plc = [1, 2, 3]
	R = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	T1 = []
	T2 = []
	T3 = []
	T4 = []
	demand = [10, 25, 100]
	for r in R:
		tt1 = []
		tt2 = []
		tt3 = []
		tt4 = []
		for i in range(20):
			SFCplacement(plc, lp, demand)
			t1 = generateTraffic(plc[0], r, "possion", 10)
			SFCplacement(plc, lp, demand)
			t2 = generateTraffic(plc[0], r, "normal30", 10)
			SFCplacement(plc, lp, demand)
			t3 = generateTraffic(plc[0], r, "normal20", 10)
			SFCplacement(plc, lp, demand)
			t4 = generateTraffic(plc[0], r, "normal10", 10)
			tt1.append(t1)
			tt2.append(t2)
			tt3.append(t3)
			tt4.append(t4)
		T1.append(sum(tt1)/len(tt1))
		T2.append(sum(tt2) / len(tt2))
		T3.append(sum(tt3) / len(tt3))
		T4.append(sum(tt4) / len(tt4))
	plt.plot(R, T1, label = "Possion Traffic", marker="v")
	plt.plot(R, T2, label=r"Normal Traffic $\delta = 30%$", marker="o")
	plt.plot(R, T3, label=r"Normal Traffic $\delta = 20%$", marker="s")
	plt.plot(R, T4, label=r"Normal Traffic $\delta = 10%$", marker="D")
	plt.xlabel("Traffic generation rate")
	plt.ylabel("End-to-end delay (s)")
	plt.grid(linestyle='--')
	plt.legend(loc='upper left', prop={'size': 7})
	plt.savefig("genRate1.eps", format='eps')

	plt.show()

def estimteNetwork(n):
	response = ''
	while response == '':
		try:
			response = requests.get("http://10.0.0.%i:3000/estimateNetwork"%(n+2))
			break
		except:
			print("Connection refused by the server..")
			print("Let me sleep for 1 seconds")
			print("ZZzzzz...")
			time.sleep(1)
			print("Was a nice sleep, now let me continue...")
			continue

	return response.content

def echoTest():
    response = requests.post("http://10.0.0.2:3000/form", json={"placement": [1,2,3]})
    return response.json()

def SFCplacement(placement, lastplacement, demand):
	start = time.time()
	response = ''
	while response == '':
		try:
			response = requests.post("http://10.0.0.%i:3000/sfc_request" % (placement[0]+1), json={"demand": demand,
																									 "placement": placement,
																									 "lastplacement": lastplacement})
			break
		except:
			print("Connection refused by the server..")
			print("Let me sleep for 1 seconds")
			print("ZZzzzz...")
			time.sleep(1)
			print("Was a nice sleep, now let me continue...")
			continue

	return time.time()-start, response

def updateEstimation(estimation, plc, T, demand):
	#print(estimation, placement, T, lam)
	placement = plc
	usr = getCurrentAp()
	placement.append(usr-1)
	for i, c in enumerate(estimation[1::2]):
		# print(estimation[1::2])
		# print(i, demand)
		T[placement[i]-1][placement[i]-1].append(random.uniform(0.75*c/demand[i], 1.25*c/demand[i]))
	for i, t in enumerate(estimation[0::2]):
		# print(i, t)
		T[placement[i-1]-1][placement[i]-1].append(random.uniform(0.75*t, 1.25*t))
	return None

def getCapacitylist():
	T = [[[] for n in range(numNodes)] for m in range(numNodes)]
	for n in range(numNodes):
		eT = eval(estimteNetwork(n))
		print(eT)
		for i in range(numNodes):
			for j in range(numNodes):
				for e in eT[i][j]:
					e = random.uniform(0.5 * e, 1.5 * e)
				T[i][j] += eT[i][j]
	return T

def getCurrentAp():
	str = subprocess.check_output(["iw", "usr-wlan0", "link"])
	return int(re.search("ssid-ap([0-9])", str).group(1))

def CMAB(iters, eps):

	# f = open("estimation.txt", 'r')
	# if f.read() == '':
	# 	T = getCapacitylist()
	# 	f = open("estimation.txt", 'w')
	# 	f.write(str(T))
	# else:
	# 	f = open("estimation.txt", 'r')
	# 	fstr = f.readline()
	# 	T = ast.literal_eval(fstr)
	# T = getCapacitylist()
	# f = open("estimation.txt", 'w')
	# f.write(str(T))
	# f.close()
	# # #print("network delay estimations:", T)

	T =[[[0.01], [0.45], [0.1], [0.25], [1.0], [0.6]],
		   [[0.45], [0.02], [0.35], [0.2], [0.95], [0.55]],
		   [[0.1], [0.35], [0.03], [0.15], [0.9], [0.5]],
		   [[0.25], [0.2], [0.15], [0.04], [0.75], [0.35]],
		   [[1.0], [0.95], [0.9], [0.75], [0.05], [0.4]],
		   [[0.6], [0.55], [0.5], [0.35], [0.4], [0.06]]]
	#T = [[[0.0] for i in range(numNodes)] for j in range(numNodes)]
	lp = random.sample([i for i in range(numNodes)], numServices)
	lpo = lp
	pos = getCurrentAp() - 1
	#lp = [pos for i in range(numServices)]
	demand = [100 for i in range(numServices)]
	Demand = []
	Pos= []
	#SFCplacement(plc, lp, demand)
	Cost = []
	AvgCost = []
	tCost = []
	offCost = []
	lam = 100
	#gCost = eGreedy(iters, Demand, 0.1)
	for it in range(iters):
		print("###################################")
		pos = getCurrentAp() - 1
		Pos.append(pos)
		Demand.append(demand)
		print("SFC demand:", demand)
		print("current position:", pos)
		#plc = random.sample([1,2,3,4], 3)
		plc, cost = DP(numNodes, T, lp, demand, pos, numServices, indexCloud, it+1, eps)
		tCost.append(cost)
		print("estimated service time:", cost)
		print("placement", plc)
		#estimateSFC(plc, c, f)
		lp = [i+1 for i in lp]
		SFCplacement(plc, lp, demand)
		lp = [i-1 for i in plc]
		estimation, t = generateTraffic(plc[0], 0.0, "possion", lam)
		print("service time:", t)
		updateEstimation(estimation, plc, T, demand)
		demand = [random.choice([10, 25, 100]) for i in range(numServices)]
		#random.shuffle(demand)
		#demand = [random.choice([10, 25, 100]) for i in range(numServices)]
		Cost.append(t)
		AvgCost.append(sum(Cost)/len(Cost))
		print("###################################")
	# f = open("estimation1000.txt", 'w')
	# f.write(str(T))

	return Cost, tCost, Demand, Pos, T

def Offline(iters, D, P):
	# f = open("Demand1.txt", 'r')
	# fstr = f.readline()
	# D = ast.literal_eval(fstr)
	# D[0] = [100, 25, 10]
	# print(D)
	#
	# f = open("Pos1.txt", 'r')
	# fstr = f.readline()
	# P = ast.literal_eval(fstr)
	# print(P)
	lp = random.sample([i for i in range(numNodes)], numServices)
	offT =[[[0.01], [0.45], [0.1], [0.25], [1.0], [0.6]],
		   [[0.45], [0.02], [0.35], [0.2], [0.95], [0.55]],
		   [[0.1], [0.35], [0.03], [0.15], [0.9], [0.5]],
		   [[0.25], [0.2], [0.15], [0.04], [0.75], [0.35]],
		   [[1.0], [0.95], [0.9], [0.75], [0.05], [0.4]],
		   [[0.6], [0.55], [0.5], [0.35], [0.4], [0.06]]]
	Cost = []
	lam = 100
	for it in range(iters):
		plc, cost = DPg(numNodes, offT, lp, D[it], P[it], numServices, indexCloud, 1)
		lp = [i-1 for i in plc]
		demand = [10, 25, 100]
		random.shuffle(demand)
		Cost.append(cost)
	return Cost

def DynamicGreedy(iters, D, eps):
	# f = open("estimation.txt", 'r')
	# if f.read() == '':
	# 	T = getCapacitylist()
	# 	f = open("estimation.txt", 'w')
	# 	f.write(str(T))
	# else:
	# 	f = open("estimation.txt", 'r')
	# 	fstr = f.readline()
	# 	T = ast.literal_eval(fstr)
	#f.close()
	lp = random.sample([i+1 for i in range(numNodes)], numServices)
	Pos= []
	Cost = []
	lam = 100
	actions = [list(a) for a in (permutations([i+1 for i in range(numNodes)], numServices))]
	print(actions)
	values = [[] for a in actions]
	for it in range(iters):
		eps = eps*(1.0 - it/iters)
		print("###################################")
		p = np.random.random()
		print(p)
		if not any(values) or p < eps:
			plc = random.sample([i+1 for i in range(numNodes)], numServices)
		else:
			m = []
			for v in values:
				if v != []:
					m.append(sum(v)/len(v))
				else:
					m.append(1000.0)
			plc = actions[np.argmin(m)]
		print("current exploration:", plc)
		SFCplacement(plc, lp, D[it])
		lp = plc
		_, t = generateTraffic(plc[0], 0.0, "possion", lam)
		values[actions.index(plc)].append(t)
		print("values:", values)
		Cost.append(t)
		print("###################################")
	return Cost

def eGreedy(iters, D, eps):
	# f = open("estimation.txt", 'r')
	# if f.read() == '':
	# 	T = getCapacitylist()
	# 	f = open("estimation.txt", 'w')
	# 	f.write(str(T))
	# else:
	# 	f = open("estimation.txt", 'r')
	# 	fstr = f.readline()
	# 	T = ast.literal_eval(fstr)
	#f.close()
	lp = random.sample([i+1 for i in range(numNodes)], numServices)
	Pos= []
	Cost = []
	lam = 100
	actions = [list(a) for a in (permutations([i+1 for i in range(numNodes)], numServices))]
	print(actions)
	values = [[] for a in actions]
	for it in range(iters):
		print("###################################")
		p = np.random.random()
		print(p)
		if not any(values) or p < eps:
			plc = random.sample([i+1 for i in range(numNodes)], numServices)
		else:
			m = []
			for v in values:
				if v != []:
					m.append(sum(v)/len(v))
				else:
					m.append(1000.0)
			plc = actions[np.argmin(m)]
		print("current exploration:", plc)
		SFCplacement(plc, lp, D[it])
		lp = plc
		_, t = generateTraffic(plc[0], 0.0, "possion", lam)
		values[actions.index(plc)].append(t)
		print("values:", values)
		Cost.append(t)
		print("###################################")
	return Cost

def adjust(G, t):
    for (n, d) in G.nodes(data=True):
        d['weight'] = d['estimate'] + 0.1*sqrt(3 * log(t) / (2 * d['time']))
    for (u, v, d) in G.edges(data=True):
        d['weight'] = d['estimate'] - 0.1*sqrt(3 * log(t) / (2 * d['time']))
        if d['weight'] < 0.0:
            d['weight'] = 0.0
    return None

def DP(n, T, lp, d, pos, service_num, indexcloud, it, eps):
	# print(service_num)
	# n = len(G.nodes)
	# print("n:", n)
	# print("T:", T)
	# print("T_1_1:", T[1][1])
	# print("c_1_1:", sum(T[1][1])/len(T[1][1]))
	c = [0.0 for i in range(n)] #
	for i in range(n):
		c[i] = sum(T[i][i])/len(T[i][i]) - eps*sqrt(3 * log(it)/(2 * len(T[i][i])))
		if c[i] < 0.0:
			c[i] = 0.0
	# for i in range(numNodes):
	# 	print(T[i][i])
	print("c:", c)
	f = [[0.0 for j in range(n)] for k in range(n)]

	for i in range(n):
		for j in range(n):
			if i == j:
				f[i][j] = 0.0
			else:
				f[i][j] = sum(T[i][j])/len(T[i][j]) - 0.05*sqrt(3 * log(it)/(2 * len(T[i][j])))
				if f[i][j] < 0.0:
					f[i][j] = 0.0
				elif f[i][j] > 1.0:
					f[i][j] = 1.0
	print("f:", f)
	# print(c)
	# print(f)
	# f = getShortestPathlist(G)
	# print(context.last_placement)
	# print(context.pos)
	Host = [[[] for i in range(n)] for i in range(service_num)]
	D = [[0.0 for i in range(n)] for i in range(service_num)]

	i = service_num - 1
	while (i >= 0):
		for j in range(n):
			minCost = inf
			if i == service_num - 1:
				# print(D)
				# print(d)
				# print(f)
				# print(lp)
				# print(i, j, pos)
				D[i][j] = d[i] * c[j] + 4*f[lp[i]][j] + f[j][pos]
			elif i == 0:
				minCost = inf
				host = 0
				for k in range(n):
					# if cloud in list(G.nodes):
						# print("cloud", list(G.nodes).index(cloud))
					if (j not in Host[i + 1][k] and j != k) or (j == indexcloud):
						# print(D)
						# print(d)
						# print(f)
						# print(lp)
						# print(i, j, k, pos)
						cost = f[pos][j] + d[i] * c[j] + 4*f[lp[i]][j] + f[j][k] + D[i+1][k]
						if minCost >= cost:
							minCost = cost
							host = k
					# else:
					# 	# print("edge")
					# 	if (j not in Host[i + 1][k] and j != k):
					# 		cost = f[pos][j] + d[i] * c[j] + f[lp[i]][j] + f[j][k] + D[i + 1][k]
					# 		if minCost >= cost:
					# 			minCost = cost
					# 			host = k
				D[i][j] = minCost
				Host[i][j] = [host] + Host[i + 1][host]
			# cost = f[pos][j] + d[i]*c[j] + f[lp[i]][j] + min(f[i][k]+D[i+1][k] for k in range(n))
			else:
				minCost = inf
				host = 0
				for k in range(n):
					# if cloud in list(G.nodes):
						# print("cloud", list(G.nodes).index(cloud))
					if (j not in Host[i + 1][k] and j != k) or (j == indexcloud):
						# print(D)
						# print(d)
						# print(f)
						# print(lp)
						# print(i, j, k, pos)
						cost = d[i] * c[j] + 4*f[lp[i]][j] + f[j][k] + D[i+1][k]
						if minCost >= cost:
							minCost = cost
							host = k
					# else:
					# 	# print("edge")
					# 	if (j not in Host[i + 1][k] and j != k):
					# 		cost = d[i] * c[j] + f[lp[i]][j] + f[j][k] + D[i + 1][k]
					# 		if minCost >= cost:
					# 			minCost = cost
					# 			host = k
				D[i][j] = minCost
				Host[i][j] = Host[i + 1][host] + [host]
			# cost = d[i]*c[j] + min(f[i][k]+D[i+1][k] for k in range(n))
		i -= 1
	m = D[0].index(min(D[0]))

	arm = [p+1 for p in [m] + Host[0][m]]
	# for a in Host[m]:
	#
	#
	# arm.append(list(G.nodes)[m])
	# for i in range(service_num-1):
	#     m = Host[i][m]
	#     arm.append(list(G.nodes)[m])
	#print(min(D[0]))
	return arm, min(D[0])

def DPg(n, T, lp, d, pos, service_num, indexcloud, it):
	c = [0.0 for i in range(n)] #
	for i in range(n):
		c[i] = sum(T[i][i])/len(T[i][i])
		if c[i] < 0.0:
			c[i] = 0.0
	f = [[0.0 for j in range(n)] for k in range(n)]

	for i in range(n):
		for j in range(n):
			if i == j:
				f[i][j] = 0.0
			else:
				f[i][j] = sum(T[i][j])/len(T[i][j])
				if f[i][j] < 0.0:
					f[i][j] = 0.0
				elif f[i][j] > 1.0:
					f[i][j] = 1.0
	Host = [[[] for i in range(n)] for i in range(service_num)]
	D = [[0.0 for i in range(n)] for i in range(service_num)]

	i = service_num - 1
	while (i >= 0):
		for j in range(n):
			minCost = inf
			if i == service_num - 1:
				D[i][j] = d[i] * c[j] + 4*f[lp[i]][j] + f[j][pos]
			elif i == 0:
				minCost = inf
				host = 0
				for k in range(n):
					if (j not in Host[i + 1][k] and j != k) or (j == indexcloud):
						cost = f[pos][j] + d[i] * c[j] + 4*f[lp[i]][j] + f[j][k] + D[i+1][k]
						if minCost >= cost:
							minCost = cost
							host = k
				D[i][j] = minCost
				Host[i][j] = [host] + Host[i + 1][host]
			else:
				minCost = inf
				host = 0
				for k in range(n):
					if (j not in Host[i + 1][k] and j != k) or (j == indexcloud):
						cost = d[i] * c[j] + 4*f[lp[i]][j] + f[j][k] + D[i+1][k]
						if minCost >= cost:
							minCost = cost
							host = k
				D[i][j] = minCost
				Host[i][j] = Host[i + 1][host] + [host]
		i -= 1
	m = D[0].index(min(D[0]))
	arm = [p+1 for p in [m] + Host[0][m]]
	return arm, min(D[0])

def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x * 1e-6)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

if __name__ == '__main__':
	if "-p" in sys.argv:
		#demand=[random.choice([2.5, 25, 100]) for i in range(int(sys.argv[2]))]
		placement = [int(arg) for arg in sys.argv[2:5]]
		# print(placement)
		lastplacement = [int(arg) for arg in sys.argv[5:8]]
		# print(lastplacement)
		print("Start SFCplacement...")
		time, reponse = SFCplacement(placement, lastplacement)
		print("placement complete, time spent: ", time)
	elif "-e" in sys.argv:
		print("start estimation on node %i"%(int(sys.argv[2])))
		time = estimateNode(int(sys.argv[2]))
		print("estimation complete, rtt: %f"%(time))
	elif "-s" in sys.argv:
		sendingPackets(int(sys.argv[2]))
	elif "-l" in sys.argv:
		print("start estimation on link %i to %i"%(int(sys.argv[2]), int(sys.argv[3])))
		time = estimatelink(int(sys.argv[2]), int(sys.argv[3]))
		print("estimation complete, link delay: %f"%(time))
	elif "-g" in sys.argv:
		trafficGenerationRate()
	elif "-c" in sys.argv:
		iters = int(sys.argv[2])
		e = sys.argv[3]
		Cost, tCost, Demand, Pos, T = CMAB(iters, 0.075)
		f = open("Demand%s.txt"%e, "w")
		f.write(str(Demand))
		f.close()
		f = open("Posd%s.txt"%e, "w")
		f.write(str(Pos))
		f.close()
		# f = open("tCostd1.txt", "w")
		# for e in tCost:
		# 	f.write('%s\n' % e)
		# f.close()
		f = open("Cost%s.txt"%e, "w")
		for c in Cost:
			f.write('%s\n' % c)
		f.close()
		f = open("estimation%s.txt"%e, "w")
		f.write(str(T))
		f.close()
		dygCost = DynamicGreedy(iters, Demand, 0.15)
		egCost = eGreedy(iters, Demand, 0.15)
		f = open("dygCost%s.txt"%e, "w")
		for c in dygCost:
			f.write('%s\n' % c)
		f.close()
		f = open("egCost%s.txt"% e, "w")
		for c in egCost:
			f.write('%s\n' % c)
		f.close()
		# m, h = mean_confidence_interval(Cost)
		# me, he = mean_confidence_interval(tCost)
		# f = open("delayvscost4.txt", "a")
		# f.write(str(m) + " " + str(h) + "\n")
		# f.write(str(me) + " " + str(he) + "\n")
		# f.close()
		# f = open("offCostd1.txt", "w")
		# for e in oCost:
		# 	f.write('%s\n'%e)
		# f.close()
		x = [i for i in range(iters)]
		#offLine = [sum(tCost)/len(tCost) for i in range(iters)]
		#print(offLine)
		#AvgEst = [sum(tCost[0:i + 1]) / len(tCost[0:i + 1]) for i in range(len(tCost))]
		AvgCost = [sum(Cost[0:i + 1]) / len(Cost[0:i + 1]) for i in range(len(Cost))]
		Avgdg = [sum(dygCost[0:i + 1]) / len(dygCost[0:i + 1]) for i in range(len(dygCost))]
		Avgeg = [sum(egCost[0:i + 1]) / len(egCost[0:i + 1]) for i in range(len(egCost))]
		plt.plot(x, AvgCost, label="measured cost")
		#plt.plot(x, AvgEst, label="expected cost")
		plt.plot(x, Avgdg, label="annealing (adaptive) egreedy")
		plt.plot(x, Avgeg, label="egreedy")
		plt.legend()
		#plt.plot(x, off, label = "offline optimum")
		plt.xlabel("Learning slot")
		plt.ylabel("Average response time (s)")
		plt.grid(linestyle='--')
		plt.savefig("learning%s.eps"%e, format='eps')
		plt.savefig("learning%s.png"%e, format='png')
		plt.show()
	elif "-en" in sys.argv:
		T = getCapacitylist()
		print(T)
	elif "-m" in sys.argv:
		# CMAB = []
		# Cloud = []
		# Edge = []
		# Offline = []
		# heuristic = []
		# for i in range(5):
		# 	G = createNetwork(5, i, 1)
		# 	cmab, cloud, edge, offline, heu = main2(G, n, th)
		# 	CMAB.append(cmab)
		# 	Cloud.append(cloud)
		# 	Edge.append(edge)
		# 	Offline.append(offline)
		# 	heuristic.append(heu)

		# CMAB = [Cost1[i]/Cost2[i] for i in range(5)]
		# Cloud = [Cost3[i]/Cost2[i] for i in range(5)]
		# Edge = [Cost4[i]/Cost2[i] for i in range(5)]
		ILP = [1.0 for i in range(5)]
		labels = ("RW", "RD", "TVC", "GM")
		x = np.arange(len(labels))
		Y = [10.1781151891, 11.3251267552, 11.4250441456, 13.2298670268]
		#width = 0.5

		# fig, ax = plt.subplots()
		# formatter = FuncFormatter(millions)
		# ax.yaxis.set_major_formatter(formatter)
		plt.bar(x, Y)
		# plt.bar(x - 3 * width / 2, CMAB, width, label='CMAB')
		# plt.bar(x - width / 2, Cloud, width, label='Cloud', hatch='-')
		# plt.bar(x + width / 2, Edge, width, label='Edge', hatch='/')
		# plt.bar(x + 3 * width / 2, heuristic, width, label='Heuristic')
		plt.ylabel('Average response time (s)')
		plt.xlabel('Mobility models')
		plt.xticks(x, labels)
		# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.legend()

		plt.savefig("mobilitymodel.eps", format='eps')
		plt.savefig("mobilitymodel.png", format='png')
		plt.show()
	# elif "-cmab" in sys.argv:
	# 	CMAB(100)
	elif "-o" in sys.argv:
		solver = 5.490934206743051
		Cost = [float(r) for r in open("Cost6.txt", 'r').readlines()]
		avgc = [sum(Cost[0:i + 1]) / len(Cost[0:i + 1]) for i in range(len(Cost))]
		off = Offline(100)
		solver = [solver for i in range(100)]
		x = [i for i in range(len(off))]
		plt.plot(x, off, label="offline")
		plt.plot(x, avgc, label="CMAB")
		plt.plot(x, solver, label="Gurobi")
		plt.xlabel("Learning slot")
		plt.ylabel("Average response time (s)")
		plt.grid(linestyle='--')
		plt.legend()
		plt.savefig("offline.eps", format='eps')
		plt.savefig("offline.png", format='png')
		plt.show()
	elif "-r" in sys.argv:
		# AvgCost, Cost, offl, tCost = CMAB(iters)
		# m, h = mean_confidence_interval(Cost)
		# me, he = mean_confidence_interval(tCost)
		# f = open("delayvscost.txt", "a")
		# f.write(str(m) + " " + str(h) + "\n")
		# f.write(str(me) + " " + str(he) + "\n")
		# f.close()

		Cost = [float(r) for r in open("Costg2.txt", 'r').readlines()]
		tCost = [float(r) for r in open("tCost500.txt", 'r').readlines()]
		oCost = [float(r) for r in open("offCost500.txt", 'r').readlines()]
		gCost = [float(r) for r in open("gCostg2.txt", 'r').readlines()]
		offline = [float(open("offline100_1.txt").read()) for i in range(len(Cost))]
		# for i, a in enumerate(AvgCost):
		# 	if i == 0:
		# 		Cost.append(a)
		# 	else:
		# 		Cost.append((i+1)*AvgCost[i] - i*AvgCost[i-1])
		x = [i for i in range(len(Cost))]
		avgc = [sum(Cost[0:i + 1]) / len(Cost[0:i + 1]) for i in range(len(Cost))]
		avgt = [sum(tCost[0:i+1]) / len(tCost[0:i+1]) for i in range(len(tCost))]
		avgg = [sum(gCost[0:i + 1]) / len(gCost[0:i + 1]) for i in range(len(gCost))]
		avgo = [sum(oCost[0:i + 1]) / len(oCost[0:i + 1]) for i in range(len(oCost))]
		# AvgCost = [sum(Cost[0:i+1]) / len(Cost[0:i+1]) for i in range(len(Cost))]
		AvgCost = Cost
		#offLine = [sum(EstCost) / len(EstCost) for i in range(len(EstCost))]
		#off = [offl for i in range(iters)]
		# print(offLine)
		plt.plot(x, avgc, label="CMAB")
		#plt.plot(x, avgt, label="expected cost")
		plt.plot(x, offline, label="offline")
		plt.plot(x, avgg, label="$\epsilon$-greedy")
		plt.legend()
		# plt.plot(x, off, label = "offline optimum")
		plt.xlabel("Learning slot")
		plt.ylabel("Average response time (s)")
		plt.ylim(bottom=0)
		plt.xlim(left=-10)
		plt.grid(linestyle='--')
		plt.legend()
		plt.savefig("replot8.eps", format='eps')
		plt.savefig("replot8.png", format='png')
		plt.show()
	elif "-c1" in sys.argv:
		C = [0.05, 0.075, 0.1, 0.125, 0.15]
		iters = 100
		x = [i for i in range(iters)]
		for c in C:
			Cost, _, _, _ = CMAB(iters, c)
			avgc = [sum(Cost[0:i + 1]) / len(Cost[0:i + 1]) for i in range(len(Cost))]
			plt.plot(x, avgc, label="c = %f"%c)
		plt.legend()
		plt.xlabel("Learning slot")
		plt.ylabel("Average response time (s)")
		plt.ylim(bottom=0)
		plt.xlim(left=-10)
		plt.grid(linestyle='--')
		plt.legend()
		plt.savefig("explorelevel2.eps", format='eps')
		plt.savefig("explorelevel2.png", format='png')
		plt.show()
	elif "-r1" in sys.argv:
		# AvgCost, Cost, offl, tCost = CMAB(iters)
		# m, h = mean_confidence_interval(Cost)
		# me, he = mean_confidence_interval(tCost)
		# f = open("delayvscost.txt", "a")
		# f.write(str(m) + " " + str(h) + "\n")
		# f.write(str(me) + " " + str(he) + "\n")
		# f.close()

		Cost = [float(r) for r in open("Cost100_1.txt", 'r').readlines()]
		tCost = [float(r) for r in open("Cost100.txt", 'r').readlines()]
		oCost = [float(r) for r in open("Cost5.txt", 'r').readlines()]
		gCost = [float(r) for r in open("gCost500.txt", 'r').readlines()]
		offline = [float(open("offline100_1.txt").read()) for i in range(len(Cost))]
		# for i, a in enumerate(AvgCost):
		# 	if i == 0:
		# 		Cost.append(a)
		# 	else:
		# 		Cost.append((i+1)*AvgCost[i] - i*AvgCost[i-1])
		x = [i for i in range(len(Cost))]
		avgc = [sum(Cost[0:i + 1]) / len(Cost[0:i + 1]) for i in range(len(Cost))]
		avgt = [sum(tCost[0:i+1]) / len(tCost[0:i+1]) for i in range(len(tCost))]
		avgg = [sum(gCost[0:i + 1]) / len(gCost[0:i + 1]) for i in range(len(gCost))]
		avgo = [sum(oCost[0:i + 1]) / len(oCost[0:i + 1]) for i in range(len(oCost))]
		# AvgCost = [sum(Cost[0:i+1]) / len(Cost[0:i+1]) for i in range(len(Cost))]
		AvgCost = Cost
		#offLine = [sum(EstCost) / len(EstCost) for i in range(len(EstCost))]
		#off = [offl for i in range(iters)]
		# print(offLine)
		plt.plot(x, avgt, label="c = 0.05")
		plt.plot(x, avgc, label="c = 0.075")
		plt.plot(x, avgo, label="c = 0.1")
		#plt.plot(x, avgg, label="$\epsilon$-greedy")
		plt.legend()
		# plt.plot(x, off, label = "offline optimum")
		plt.xlabel("Learning slot")
		plt.ylabel("Average response time (s)")
		plt.ylim(bottom=0)
		plt.xlim(left=-10)
		plt.grid(linestyle='--')
		plt.legend()
		plt.savefig("replot6.eps", format='eps')
		plt.savefig("replot6.png", format='png')
		plt.show()
