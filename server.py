from flask import Flask, request, redirect, url_for
from flask import jsonify
import netifaces as ni
import json
import requests
import time 
import sys
import random
import socket
#from socket import *
import thread
from threading import Thread
import numpy

def getMyid():
	return int(list(ni.interfaces()[1])[1])

switch_weight=2
MESSAGE = "c"
numNodes = 6
TCP_IP = '0.0.0.0'
TCP_PORT = 5005
BUFFER_SIZE = 1024
placement = []
lastplacement = []
demand = 0
nextid = 0
id = 0
i = 0
comp = random.uniform(0.01, 0.05)
usrip = "10.0.0.1"

if "-e" in sys.argv:
	comp = random.uniform(0.5, 1)
elif "-c" in sys.argv:
	comp = 0.01*(getMyid())
elif "-c1" in sys.argv:
	comp = 0.01*(getMyid()) + 0.05
elif "-c2" in sys.argv:
	comp = 0.01*(getMyid()) + 0.1
elif "-c3" in sys.argv:
	comp = 0.01 * (getMyid()) + 0.15
elif "-c4" in sys.argv:
	comp = 0.01 * (getMyid()) + 0.2


app = Flask(__name__)
def getMyip(): 
	return ni.ifaddresses(ni.interfaces()[1])[ni.AF_INET][0]['addr']



def forward(n):
	requests.post("http://10.0.0.%i:3000/form"%(n+1), json={"name": "lei"})

def processsing(demand, computation):
	print("processsing.....")
	time.sleep(demand/computation)

def switchfrom(n):
	print("switch from node%i......."%n)
	if n == getMyid():
		print("No need to switch!")
	else:
		response = ''
		i = 0
		while response == '' or i < switch_weight:
			i += 1
			try:
				response = requests.get("http://10.0.0.%i:3000/switch"%(n+1))
				break
			except:
				print("Connection refused by the server..")
				print("Let me sleep for 5 seconds")
				print("ZZzzzz...")
				time.sleep(5)
				print("Was a nice sleep, now let me continue...")
				continue
		print(response.content)
		print(response.status_code)
# @app.route('/form', methods=['POST'])
# def echoResponse():
#     json_data = request.get_json()
#     name = json_data['name']
#     return jsonify({'message': "hello %s"%name})
#     #return """Hello, {}""".format(name)


@app.route('/sfc_request', methods=['POST'])
def Response():
	json_data = request.get_json()
	#demand = json_data['demand']
	usrip = request.remote_addr
	placement = json_data['placement']
	lastplacement = json_data['lastplacement']

	id = getMyid()
	i = placement.index(id)
	demand = json_data['demand'][i]
	switchfrom(lastplacement[i])
	#print("index:", i)
	print("current processing node: ", id)
	print("current placement: ", placement)
	print("current computation ability: ", comp)
	print("current demand:", demand)
	#print("current demand: ", demand[i])
	#processsing(demand[i], comp)
	if id != placement[-1]:
		#print("next node:", placement[i+1])
		nextid = placement[i+1]
		TCPServer(nextid, usrip, demand).start()
		return redirect("http://10.0.0.%i:3000/sfc_request"%(nextid+1), code=307)
	else:
		TCPServer(None, usrip, demand).start()
		return jsonify({'message': "placement complete"})


	# return jsonify({'message': "hello %s"%name})
    #return """Hello, {}""".format(name)


@app.route('/estimate_request', methods=['GET'])
def ResponseEstimation():
	usrip = request.remote_addr
	TCPServer("usr", usrip).start()
	return "ok!"

@app.route('/switch', methods=['GET'])
def Switch():
	#processsing(float(sys.argv[2]))
	return "switch complete!"

@app.route('/estimateNetwork', methods=['GET'])
def estimateNetwork():
	T = [[[] for n in range(numNodes)] for m in range(numNodes)]
	lam = 101
	for n in range(numNodes):
		if n+1 != getMyid():
			SFCplacement([n+1], [n+1], [100])
			estimation, _ = generateTraffic(n+1, 0.0, "possion", lam)
			updateEstimation(estimation, [n+1], T, lam)
	return str(T)

def parseRet(str):
	strlist = str.split()
	timelist = strlist[1:]
	delaylist = []
	for i, s in enumerate(timelist[:-1]):
		delaylist.append(float(timelist[i+1]) - float(timelist[i]))
	return delaylist

def updateEstimation(estimation, placement, T, lam):
	usr = getMyid()
	placement.append(usr)
	for i, c in enumerate(estimation[1::2]):
		T[placement[i]-1][placement[i]-1].append(c/lam)
	for i, t in enumerate(estimation[0::2]):
		T[placement[i-1]-1][placement[i]-1].append(t)
	return None

def SFCplacement(placement, lastplacement, demand):
	start = time.time()
	response = ''
	while response == '':
		try:
			response = requests.post("http://10.0.0.%i:3000/sfc_request" % (placement[0] + 1), json={"demand": demand,
																									 "placement": placement,
																									 "lastplacement": lastplacement})
			break
		except:
			print("Connection refused by the server..")
			print("Let me sleep for 5 seconds")
			print("ZZzzzz...")
			time.sleep(5)
			print("Was a nice sleep, now let me continue...")
			continue

	return time.time() - start, response

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
	print("Time stamps:", parseRet(ret))
	s.shutdown(socket.SHUT_WR)
	s.close()
	t = time.time()-start
	#print("complete, rtt:", t)
	return parseRet(ret), t
# def TCPServer(placement, usrip, i, id):
# 	# if id != placement[0]:
# 	# 	TCP_IP = "10.0.0.%i"%(placement[i-1]+1)
# 	# else:
# 	# 	TCP_IP = usrip
# 	print(TCP_IP)
# 	print("runing TCPserver.....")
# 	s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 	s1.bind((TCP_IP, TCP_PORT))
# 	s1.listen(1)
#
# 	conn, addr = s1.accept()
# 	print('Connection address:', addr)
# 	D = []
# 	while 1:
# 		data = conn.recv(BUFFER_SIZE)
# 		if not data: break
# 		print("received data:", data)
# 		D.append(data)
# 		#conn.send(data)
# 	conn.close()
# 	s1.close()
# 	if id != placement[-1]:
# 		s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 		s2.connect("10.0.0.%i"%(nextid+1), TCP_PORT)
# 		for d in D:
# 			s2.send(data)
# 	else:
# 		s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 		s2.connect(usrip, TCP_PORT)

class TCPServer(Thread):
	def __init__(self, nextid, usrip, demand):
		Thread.__init__(self)
		self.nextid = nextid
		self.usrip = usrip
		self.demand = demand

	def	run(self):
		print(TCP_IP)
		print("runing TCPserver.....")
		s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s1.bind((TCP_IP, TCP_PORT))
		s1.listen(1)

		conn, addr = s1.accept()
		print('Connection address:', addr)

		D = []
		start = time.time()
		data = conn.recv(1)
		recvtime = time.time()
		i = 0
		while i <= self.demand:
			data = conn.recv(1)
			time.sleep(comp)
			D.append(data)
 			# if "-ec" in sys.argv:
			# 	endata = data.encode('base64', 'strict')
			# 	D.append(endata)
			# 	print("decoded data:", endata)
			# elif "-dc" in sys.argv:
			# 	dedata = data.decode()
			# 	D.append(dedata)
			# 	print("encoded data:", dedata)
			# elif "-f" in sys.argv:
			# 	f = open("payload.txt", "a")
			# 	f.write(data)
			# 	f.close()
			i += 1
		while True:
			data = conn.recv(1)
			D.append(data)
			if not data: break
		D.append(' ' + str(recvtime))
		pt = time.time() - start
		print("processing done......")
		s1.shutdown(socket.SHUT_WR)
		s1.close()
		# if self.nextid == "usr":
		# 	for d in D:
		# 		conn.send(d)
		# 	conn.send(str(pt))
		# 	conn.close()
		# 	s1.close()
		# 	return None
		# else:
		# 	conn.send("ok")
		# 	conn.close()
		# 	s1.close()
		# conn.send(data)
		#print("nextid:", self.nextid)
		if self.nextid != None:
			print("Sending to %i"%self.nextid)
			s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			s2.connect(("10.0.0.%i" % (self.nextid+1), TCP_PORT))
			processtime = time.time()
			D.append(' ' + str(processtime))
			for d in D:
				s2.send(d)
			s2.close()
		else:
			s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			s2.connect((self.usrip, TCP_PORT))
			processtime = time.time()
			D.append(' ' + str(processtime))
			ret = ''
			for d in D:
				ret = ret+d
			s2.send(ret[-1024:-1])
			s2.close()









if __name__ == '__main__':
	if "-e" in sys.argv:
		print("Running on edge......")
	elif "-c" in sys.argv:
		print("Running on cloud......")
	print("current computation:", comp)
	app.run(host=getMyip(), debug=True, port=3000)
	# t1 = threading.Thread(target=app.run(host=getMyip(),debug=True, port=3000))
	# t2 = threading.Thread(target=TCPServer())
	#
	# t2.start()
	# t1.start()
	# if "-e" in sys.argv:
	# 	print("Running on edge......")
	# elif "-c" in sys.argv:
	# 	print("Running on cloud......")


	# thread = Thread(target=TCPServer(placement, usrip, i, id))
	# thread.start()
	# thread = Thread(target=app.run(host=getMyip(),debug=True, port=3000))
	# thread.start()
	# TCPServer()




