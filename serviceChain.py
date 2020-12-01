import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from math import *
from itertools import permutations
from gurobipy import *
import scipy.stats
import ast

service_num = 3
cloud = (100, 100)
estimate_offset_n = 0.499
estimate_offset_e = 0.499
switch_weight = 3.0
explore_ratio = 0.01

def offline_sc(G, T, Lastplacement, Pos, Demand):
    closeArms = {}
    for node in G.nodes:
        cn = []
        for n in G.nodes:
            if nx.shortest_path_length(G, n, node) <= 2 and nx.shortest_path_length(G, n, node, weight="weight") <= 10:
                 cn.append(n)
        closeArms[node] = list(permutations(cn, service_num))


   # print(len(closeArms))

    prev = [{} for t in T]
    cost = [{} for t in T]
    avgCost = []
    costArray = []
    for t in T:
        # data = ServiceChainData(pos=Pos[t])
        if t == 0:
            for a in closeArms[Pos[t]]:
                #print(i, a)
                prev[t][a] = [Lastplacement]
                context = contextData()
                context.update(pos=Pos[t], demand=Demand[t],
                               lastplace=Lastplacement)
                cost[t][a], T, S, C = scCost(a, G, context)
        else:
            for current_arm in closeArms[Pos[t]]:
                min_cost = 10000.0
                for previous_arm in closeArms[Pos[t-1]]:
                    context.update(Pos[t], Demand[t], previous_arm)  # contextual vector
                    c, T, S, C = scCost(current_arm, G, context)
                    #print(c, T, S, C)
                    #ca = cost_array_of(b, i, u)
                    #print(ca)
                    if c < min_cost:
                        min_cost = c
                        min_arm = previous_arm

                prev[t][current_arm] = [prev[t-1][previous_arm], min_arm]
                cost[t][current_arm] = min_cost + cost[t-1][previous_arm]
        mincost = min(cost[t].values())

        avgCost.append(mincost/(t+1))
    return avgCost
# explore oracle search range
def explore_search_range():
    G = nx.grid_2d_graph(5, 5)
    for (u, d) in G.nodes(data=True):
        d['weight'] = random.uniform(1, 1.5)
        d['estimate'] = d['weight']
        d['time'] = 1
    for (u, v, d) in G.edges(data=True):
        d['weight'] = random.uniform(0.5, 1)
        d['estimate'] = d['weight']
        d['time'] = 1


    # G.add_node(cloud, weight=10, time=1, estimate=10)
    # for n in G.nodes:
    #     G.add_edge(n, cloud)
    # for (u, v, d) in G.edges(data=True):
    #     if v == cloud:
    #         d['weight'] = random.uniform(5, 10)
    #         d['estimate'] = d['weight']
    #         d['time'] = 1


    pos = random.choice(list(G.nodes)-cloud)
    lps = []
    for n in G.nodes:
        if nx.shortest_path_length(G, n, pos) <= 2 and nx.shortest_path_length(G, n, pos, weight="weight") <= 2:
            lps.append(n)
    context = contextData()
    context.update(pos=pos, demand=[random.choice([2.5, 5, 7.5]) for i in range(service_num)],
                   lastplace=[random.choice(list(lps)) for i in range(service_num)])
    Cost = []
    Time = []
    for r in range(1, 11):
        start_time = time.time()
        placement = oracle(G, context, 1, r)
        end_time = time.time()
        cost, T, S, C = scCost(placement, G, context)
        #print(T, S, C)
        Cost.append(cost)
        Time.append(end_time-start_time)
    x = [i for i in range(1, 11)]
    plt.plot(x, Cost, label='service cost')
    plt.plot(x, Time, label='time cost')
    plt.xlabel("Search range")
    plt.ylabel("Cost")
    plt.set_ylim(ymin = 0)
    plt.legend()
    plt.savefig("explore4.eps", format="eps")
    plt.show()
def dijsktraSFC(G, context):
    lp = context.last_placement
    src = context.pos
    demand = context.demand
    f = getShortestPathlist(G)
    c = getCapacitylist(G)
    H, replica = expand(G, len(demand), src)
    # des = replica[src][len(demand)-1]
  #  print("src", src)

    Q = []
    dist = {}
    prev = {}
    for n in H.nodes:
        prev[n] = []
    for v in H.nodes:
        dist[v] = 10000.0
        Q.append(v)

    dist[src] = 0.0
   # print("dist", dist)
    s = 0
    ln = (100, 100)
    while len(Q) != 0:
        u = min(Q, key=lambda k:dist[k])
        for k in replica.keys():
            if u in replica[k] or u == k:
                ou = k
                break
       # print("Q, u", Q, u)
        Q.remove(u)
        for v in H.neighbors(u):
            for k in replica.keys():
                if v in replica[k]:
                    ov = k
                    break
            if u != src:
                if replica[ov].index(v) == replica[ou].index(u)+1:



                    #print(dist[u], nx.get_edge_attributes(H, "weight")[(u, v)], nx.get_node_attributes(H, "weight")[v])
                    alt = dist[u] + nx.get_edge_attributes(H, "weight")[(u, v)]+demand[len(prev[u])]/nx.get_node_attributes(H, "weight")[v]+migrate(G, v, lp, s, replica)
                   # print("alt", alt)
                    if alt < dist[v] and ov not in prev[u]:
                        dist[v] = alt
                        prev[v]=prev[u]+[ou]
            else:
                alt = dist[u] + nx.get_edge_attributes(H, "weight")[(u, v)] + demand[len(prev[ou])] / \
                      nx.get_node_attributes(H, "weight")[v] + migrate(G, v, lp, s, replica)
                if alt < dist[v] and ov not in prev[u]:
                    dist[v] = alt
                    prev[v]=prev[u]+[ou]
    l = []
    for k in replica.keys():
        l.append(replica[k][-1])
    u = min(l, key=lambda k:dist[k])
    for k in replica.keys():
        if u in replica[k] or u == k:
            ou = k
    #print(dist)
   # print(prev)
    return prev[u][1:]+[ou], dist[u]
def C(i, j, context, c, f, G):
    lp = context.last_placement
    d = context.demand
    pos = context.pos
    if i == 0:
        return d[i]*c[j]+f[list(G.nodes).index(lp[i])][j]+min([f[j][k]+C(i+1, k, context, c, f, G) for k in range(len(c))])+f[list(G.nodes).index(pos)][j]
    elif i == service_num-1:
        return d[i]*c[j]+f[list(G.nodes).index(lp[i])][j]+f[j][list(G.nodes).index(pos)]
    else:
        #print(i, j)
        return d[i]*c[j]+f[list(G.nodes).index(lp[i])][j]+min([f[j][k]+C(i+1, k, context, c, f, G) for k in range(len(c))])
def offline(G, T, Demand, Pos, f, c, lp):

    M = [i for i in range(len(c))]
    S = [i for i in range(service_num)]
    T = [i+1 for i in range(T)]
    print("Demand:", Demand)
    print("M, S, T:", M, S, T)
    print("location:", Pos)
    print("f:", f)
    print("c:", c)
    print("last placement",lp)
    m = Model()
    x = m.addVars(len(M), len(S), len(T)+ 1, vtype=GRB.BINARY, name="x")
    xx = m.addVars(len(M), len(M), len(S), len(T)+ 1, vtype=GRB.BINARY, name="xx")
    xt = m.addVars(len(M), len(M), len(S), len(T)+ 1, vtype=GRB.BINARY, name="xt")
    for s in S:
        for i in M:
            if i == lp[s]:
                m.addConstr(x[i, s, 0] == 1)
            else:
                m.addConstr(x[i, s, 0] == 0)
    # for s in S:
    #     for t in range(len(T)+1):
    #         for n in range(len(G.nodes)):
    #             m.addVar(0.0, 1.0, 1.0, vtype=GRB.BINARY, name="x_%i,%i,%i"%(n, s, t))
    for t in T:
        for s in S[:-1]:
            for i in M:
                for j in M:
                   # m.addVar(0.0, 1.0, 1.0, vtype=GRB.BINARY, name="xx_%i,%i,%i->%i,%i" % (i, j, s, s+1, t))
                    #m.addConstr(xx[i, j, s, t] == x[i, s, t]*x[j, s+1, t])
                    m.addConstr(xx[i, j, s, t] <= x[i, s, t])
                    m.addConstr(xx[i, j, s, t] <= x[j, s+1, t])
                    m.addConstr(xx[i, j, s, t] >= x[i, s, t] + x[j, s+1, t] - 1)
    for t in T:
        for s in S:
            for i in M:
                for j in M:
                   # m.addVar(0.0, 1.0, 1.0, vtype=GRB.BINARY, name="xt_%i,%i,%i,%i->%i " % (i, j, s, t-1, t))
                    #m.addConstr(xt[i, j, s, t] == x[i, s, t-1]*x[j, s, t])
                    m.addConstr(xt[i, j, s, t] <= x[i, s, t-1])
                    m.addConstr(xt[i, j, s, t] <= x[i, s, t])
                    m.addConstr(xt[i, j, s, t] >= x[i, s, t-1] + x[i, s, t] - 1)


    #print(x)


    for t in T:
        for s in S:
            m.addConstr(quicksum(x[n, s, t] for n in M) == 1)

    # #m.addConstrs((quicksum(x[n,s,t] for n in M) == 1 for s in S)for t in T)
    for t in T:
        for n in M[:-1]:
            m.addConstr(quicksum(x[n, s, t] for s in S) <= 1)
        m.addConstr(quicksum(x[M[-1], s, t] for s in S) <= 3)
    #m.addConstrs((quicksum(x[n,s,t] for s in S) <= 1 for n in M) for t in T)

    # for t in T:
    #     for s in S:
    #         sum1 = 0
    #         for n in M:
    #             sum1 += x[n, s, t]
    #         m.addConstr(sum1 == 1)
    # for t in T:
    #     for n in M:
    #         sum2 = 0
    #         for s in S:
    #             sum2 += x[n, s, t]
    #         m.addConstr(sum2 <= 1)
    #         m.addConstr(sum2 <= 1)
    # for t = T:
    #     for s in S:
    #         s = 0
    #         for n in M:
    #             s += x[]
    #for t in T:
    # m.setObjective(quicksum(quicksum((Demand[t][s]*(quicksum(x[i, s, t]* c[i] for i in M))) for s in S)
    #                 + quicksum(quicksum(quicksum(f[i][j]*x[i, s, t]*x[j, s+1, t] for j in M) for i in M) for s in S[:-1])
    #                 + quicksum(x[i, 0, t]*f[Pos[t]][i] for i in M) + quicksum(x[i, S[-1], t]*f[i][Pos[t]] for i in M)
    #                 + quicksum(quicksum(quicksum(f[j][i]*x[j,s,t-1]*x[i,s,t] for i in M) for j in M) for s in S) for t in T),
    #                 GRB.MINIMIZE)


    m.setObjective(quicksum(quicksum((Demand[t][s]*(quicksum(x[i, s, t]* c[i] for i in M))) for s in S)
                    + quicksum(quicksum(quicksum(f[j][i]*xx[i, j, s, t] for i in M) for j in M) for s in S[:-2])
                    + quicksum(x[i, 0, t]*f[Pos[t]][i] for i in M) + quicksum(x[i, S[-1], t]*f[i][Pos[t]] for i in M)
                    + quicksum(quicksum(quicksum(switch_weight*f[j][i]*xt[i,j,s,t] for i in M) for j in M) for s in S) for t in T),
                    GRB.MINIMIZE)
    # m.write("model.lp")
    m.optimize()
    for v in m.getVars():
        if v.x > 0:

            print('%s %g' % (v.varName, v.x))

    return m.objVal
def expand(G, l, src):
    replica = {u:[] for u in G.nodes}
    H = nx.DiGraph()
    H.add_node(src)
   # print(list(G.nodes(data=True)))
    for (u, d) in list(G.nodes(data=True)):
       # print(u, d)
        for i in range(l):
            H.add_node(str(u)+"(%i)"%i, weight=d["weight"])
            replica[u].append(str(u)+"(%i)"%i)
    for (u, v, d) in list(G.edges(data=True)):
       # print(u, v, d)
        if u == src:
            H.add_edge(u, replica[v][0], weight = d["weight"])
            H.add_edge(u, replica[u][0], weight = 0)
        elif v == src:
            H.add_edge(v, replica[u][0], weight=d["weight"])
            H.add_edge(v, replica[v][0], weight=0)
        for i in range(l-1):
          #  print(replica[u][i], replica[v][i+1])
          #  print(replica[v][i], replica[u][i + 1])
            H.add_edge(replica[u][i], replica[v][i+1], weight = d["weight"])
            H.add_edge(replica[v][i], replica[u][i+1], weight = d["weight"])
    # for e in H.edges:
    #   #  print(e)
    return H, replica

    #if u != src:
def generateGridwithWeight(width, length, linkdelay, capacity):
    G = G = nx.grid_2d_graph(width, length)
    for (u, d) in G.nodes(data=True):
        d['weight'] = capacity
    for (u, v, d) in G.edges(data=True):
        d['weight'] = linkdelay
    return G
class contextData(object):
    def __init__(self):
        self.discount = [0.3, 0.3, 0.3]
        self.demand = []
        self.pos = ()
        self.last_placement = []


    def update(self, pos, demand, lastplace):
        self.pos = pos
        self.demand = demand
        self.last_placement = lastplace
_G = nx.grid_2d_graph(5, 5)
def init_Estimate(G):
    for (n, d) in G.nodes(data=True):
        d['weight'] = 0
        d['time'] = 0
    for (u, v, d) in G.edges(data=True):
        d['weight'] = 0
        d['time'] = 0
    return G
# print(init_Estimate(_G).nodes(data=True))


# oracle for choosing a super arm
def oracle(G, context, t, Range):
    pos = context.pos
    n_ = []
    e_ = []
    for (n, d) in G.nodes(data=True):
        n_.append(d['weight'])
        n_.append(d['weight'])
    for (u, v, d) in G.edges(data=True):
        e_.append(d['weight'])
    adjust(G, t)
    # for (u,v,d) in G.edges(data=True):
    #     print(d)
    closeArms =[]

    for n in G.nodes:
        # print(n, pos)
        # print(nx.shortest_path_length(G, pos, n))
        if nx.shortest_path_length(G, n, pos)<= Range and nx.shortest_path_length(G, n, pos, weight="weight") <= 10:
            closeArms.append(n)
    # print(closeArms, pos)
    # for i in range(service_num-1):
    #     closeArms.append(cloud)
    # superArms = permutations(G.nodes, service_num)
    superArms = permutations(closeArms, service_num)
    # print([s for s in superArms])
    # # sl = list(superArms)
    # # #print superArm set
    # # for sa in sl:
    # #     print(sa)

    costs = []
    minCost = 10000.0
    min_arm = ()

    for s in superArms:
        scc, T, S, C = scCost(s, G, context)
        costs.append(scCost(s, G, context))
        if minCost > scc:
            minCost = scc
            min_arm = s
    resume(G, n_, e_)

    return min_arm
def oracle_ne(G, context, t):
    n_ = []
    e_ = []
    for (n, d) in G.nodes(data=True):
        n_.append(d['weight'])
    for (u, v, d) in G.edges(data=True):
        e_.append(d['weight'])
    adjust(G, t)
    minarm, cost = dijsktraSFC(G, context)
    print(scCost(minarm, G, context), cost)
    resume(G, n_, e_)

    return minarm
def oracle_dp(G, context, t):
    #print(service_num)
    n_ = []
    e_ = []
    pos = context.pos
    for (n, d) in G.nodes(data=True):
        n_.append(d['weight'])
    for (u, v, d) in G.edges(data=True):
        e_.append(d['weight'])
    adjust(G, t)

    arm, cost = DP(G, context)
    # print(arm, cost)
    # print(scCost(arm, G, context))
    # closeArms = []
    # for n in G.nodes:
    #     if nx.shortest_path_length(G, n, pos) <= 2 and nx.shortest_path_length(G, n, pos, weight="weight") <= 10:
    #         closeArms.append(n)
    # superArms = permutations(closeArms, service_num)
    # costs = []
    # minCost = 10000.0
    # for s in superArms:
    #     scc, T, S, C = scCost(s, G, context)
    #     costs.append(scCost(s, G, context))
    #     if minCost > scc:
    #         minCost = scc
    #         min_arm = s
    # print(min_arm, minCost)

    resume(G, n_, e_)

    return arm, cost
def oracle_greedy(G, context, t):
    n_ = []
    e_ = []
    pos = context.pos
    for (n, d) in G.nodes(data=True):
        n_.append(d['weight'])
    for (u, v, d) in G.edges(data=True):
        e_.append(d['weight'])
    adjust(G, t)

    candidates = []
    for n in G.nodes:
        candidates.append(n)
    #print(service_num)
    superArms = permutations(candidates, service_num)
    costs = []
    minCost = 10000.0
    for s in superArms:
        scc, T, S, C = scCost(s, G, context)
        costs.append(scCost(s, G, context))
        if minCost > scc:
            minCost = scc
            min_arm = s
    resume(G, n_, e_)

    return min_arm
def oracle_cloud():
    return [cloud, cloud, cloud]
def oracle_edge(G, context, t):
    n_ = []
    e_ = []
    for (n, d) in G.nodes(data=True):
        n_.append(d['weight'])
    for (u, v, d) in G.edges(data=True):
        e_.append(d['weight'])
    adjust(G, t)
    arm, cost = DP_edge(G, context)

    resume(G, n_, e_)

    return arm
def oracle_optimal(G, context):
    arm, cost = DP(G, context)
    return arm, cost
    # pos = context.pos
    # closeArms =[]
    # for n in G.nodes:
    #     if nx.shortest_path_length(G, n, pos) <= Range and nx.shortest_path_length(G, n, pos, weight="weight") <= 10:
    #         closeArms.append(n)
    #
    # # for i in range(service_num-1):
    # #     closeArms.append(cloud)
    # # superArms = permutations(G.nodes, service_num)
    # superArms = permutations(closeArms, service_num)
    # costs = []
    # minCost = 10000.0
    # min_arm = ()
    # for s in superArms:
    #     scc, T, S, C = scCost(s, G, context)
    #     costs.append(scCost(s, G, context))
    #     if minCost > scc:
    #         minCost = scc
    #         min_arm = s
    # return min_arm
def migrate(G, target, lp, i, replica):
    for k in replica.keys():
        if target in replica[k]:
            target = k
    return nx.shortest_path_length(G, lp[i], target, weight='weight')
# Calculate the cost of a selected service chain
def scCost(path, G, context):
    C = 0
    T = 0
    S = 0
    lp = context.last_placement
    dm = context.demand
    pt = list(path)
    pt.append(context.pos)
    # print(path)
    for i in range(service_num):
        node = pt[i]
        for (n, d) in G.nodes(data=True):
            if n == node:
                c = dm[i]*d['weight']
                #d['time'] = d['time'] + 1
        if lp != []:
            # if nx.shortest_path_length(G, lp[i], pt[i], weight='weight') == 0:
            #     s = 0
            # else:
            s = switch_weight*nx.shortest_path_length(G, lp[i], pt[i], weight='weight')
        else:
            s = 0
        C = C + c
        S = S + s

    for i in range(service_num+1):
        t = nx.shortest_path_length(G, pt[i-1], pt[i], weight='weight')
        #print(t)
        #P = nx.shortest_path(G, pt[i-1], pt[i], weight='weight')
        # for i in range(len(P) - 1):
        #     for (u, v, d) in G.edges(data=True):
        #         if (u==P[i] and v==P[i+1]):
        #             d['time'] = d['time'] + 1
        T = T+t

    # if S == 0.0:
    #     print("T:", T, "S:", S, "C:", C)

    return T+S+C, T,S,C
def updateEstimations(path, G, context):
    pt = list(path)
    pt.append(context.pos)
    for i in range(service_num):
        node = path[i]
        for (n, d) in G.nodes(data=True):
            if n == node:
                #print(n, d['time'], d['estimate'], random.uniform(d['weight'] - d['weight']/2.0, d['weight'] + d['weight']/2.0))
                d['time'] = d['time'] + 1
                d['estimate'] = (d['estimate'] * (d['time'] - 1) + random.uniform(d['weight'] - d['weight'],
                                    d['weight'] + d['weight'])) / d['time']
                #print(d['estimate'])
    for i in range(service_num + 1):
        if (pt[i] == cloud or pt[i-1] == cloud) and (pt[i] != pt[i-1]):
            for (u, v, d) in G.edges(data=True):
                if (u == pt[i-1] and v == pt[i]):
                    d['time'] = d['time'] + 1
                    d['estimate'] = (d['estimate'] * (d['time'] - 1) + random.uniform(
                        d['weight'] - d['weight'],
                        d['weight'] + d['weight'])) / d['time']
            # if pt[i+1] != cloud:
            #     for (u, v, d) in G.edges(data=True):
            #         if (u == pt[i] and v == pt[i+1]):
            #             d['time'] = d['time'] + 1
            #             d['estimate'] = (d['estimate'] * (d['time'] - 1) + random.uniform(
            #                 d['weight'] - d['weight'],
            #                 d['weight'] + d['weight'])) / d['time']
        else:
            t = nx.shortest_path_length(G, pt[i - 1], pt[i], weight='weight')
            P = nx.shortest_path(G, pt[i - 1], pt[i], weight='weight')
            for i in range(len(P) - 1):
                for (u, v, d) in G.edges(data=True):
                    if (u == P[i] and v == P[i + 1]):
                        d['time'] = d['time'] + 1
                        d['estimate'] = (d['estimate'] * (d['time'] - 1) + random.uniform(
                            d['weight'] - d['weight'],
                            d['weight'] + d['weight'])) / d['time']
# apply the exploration adjustment term to each estimation we got
def adjust(G, t):
    N =[]
    L = []
    for (n, d) in G.nodes(data=True):
        N.append(d['estimate'])
    for (u, v, d) in G.edges(data=True):
        L.append(d['estimate'])

    sdl = np.std(np.array(L))
    sdn = np.std(np.array(N))
    #print(explore_ratio)
    for (n, d) in G.nodes(data=True):
        d['weight'] = d['estimate'] - sdn#explore_ratio*sqrt(3 * log(t) / (2 * d['time']))
        if d['weight'] < 0.0:
            d['weight'] = 0.0
        #print(d['weight'])
    for (u, v, d) in G.edges(data=True):
        d['weight'] = d['estimate'] - sdl#explore_ratio*sqrt(3 * log(t) / (2 * d['time']))
        if d['weight'] < 0.0:
            d['weight'] = 0.0
        #print(d['weight'])

    return None
def resume(G, n_, e_):
    for i, (n, d) in enumerate(G.nodes(data=True)):
        d['weight'] = n_[i]
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        d['weight'] = e_[i]
    return None
def move(pos, G):
    P = []
    for p in G.nodes:
        if nx.shortest_path_length(G, pos, p) <= 1 and p != cloud:
            P.append(p)

    return random.choice(P)
def tryArms(G):
    pos = random.choice(list(G.nodes))
    for i in range(len(G.nodes)+len(G.edges)):
        i = i + 1
        superArm = random.sample(G.nodes, service_num)
        updateEstimations(superArm, G, pos)
        pos = move(pos, G)
    return i
def estimateArms(G):
    for (n,d) in G.nodes(data=True):
        d['time'] = d['time'] + 1
        d['estimate'] = (d['estimate'] * (d['time'] - 1) + random.uniform(d['weight'] - d['weight'],
                                                                          d['weight'] + d['weight'])) / d['time']
    for (u,v,d) in G.edges(data=True):
        d['time'] = d['time'] + 1
        d['estimate'] = (d['estimate'] * (d['time'] - 1) + random.uniform(d['weight'] - d['weight'],
                                                                          d['weight'] + d['weight'])) / d['time']
def getShortestPathlist(G):
    f = [[0.0 for m in G.nodes] for n in G.nodes]
    for i, n in enumerate(G.nodes):
        for j, m in enumerate(G.nodes):
            # if i == j:
            #     f[i][j] = inf
            # else:
            f[i][j] = nx.shortest_path_length(G, n, m, weight="weight")
    return f
def getShortestPathlist2(G):
    f = [[0.0 for m in G.nodes] for n in G.nodes]
    for i, n in enumerate(G.nodes):
        for j, m in enumerate(G.nodes):
            if i == j:
                f[i][j] = inf
            else:
                f[i][j] = nx.shortest_path_length(G, n, m, weight="estimate")
    return f
def getCapacitylist(G):
    c = [0.0 for n in G.nodes]
    for i, (n, d) in enumerate(G.nodes(data = True)):
        c[i] = d['weight']
    return c
def getCapacitylist2(G):
    c = [0.0 for n in G.nodes]
    for i, (n, d) in enumerate(G.nodes(data = True)):
        c[i] = 1.0/d['estimate']
    return c
def createNetwork(n, transLevel, compLevel):
    labelsn = {}
    labelse = {}
    nodel = []
    edgel = []

    # #create a star topology
    # G = nx.star_graph(25)
    # for (u, d) in G.nodes(data=True):
    #     d["weight"] = round(random.uniform(1, 1.5), 5)
    #     labelsn[u] = d["weight"]
    #     d['estimate'] = 0
    #     d['time'] = 0
    #
    # for (u, v, d) in G.edges(data=True):
    #     d['weight'] = round(random.uniform(0.5, 1), 5)
    #     labelse[(u, v)] = d["weight"]
    #     d['estimate'] = 0
    #     d['time'] = 0
    #     edgel.append((u,v))
    #
    # G.add_node("Cloud", weight=10, time=0, estimate=0)
    # G.add_edge("Cloud", 0, weight=random.uniform(5, 10), time=0, estimate=0)

    # labels = {}
    # for i, n in enumerate(S):
    #     labels[n] = "%i" % (i + 1)
    # # labels["Cloud"] = "Cloud"
    # labels[0] = "vSwitch"
    # posS = nx.spring_layout(S)
    # nx.draw(G, posS, labels=labels, with_labels=True)
    # nx.draw_networkx_nodes(G, posS, nodelist=["Cloud"], node_color='r', node_size=2000)
    # nx.draw_networkx_nodes(G, posS, nodelist=[0], node_color='y', node_size=2000)
    # plt.show()

    # create a grid network
    G = nx.grid_2d_graph(n, n)
    for (u, d) in G.nodes(data=True):
        if compLevel == 0:
            d['weight'] = round(random.uniform(0.2, 0.3), 5)
        elif compLevel == 1:
            d['weight'] = round(random.uniform(0.3, 0.4), 5)
        elif compLevel == 2:
            d['weight'] = round(random.uniform(0.4, 0.5), 5)
        elif compLevel == 3:
            d['weight'] = round(random.uniform(0.5, 0.6), 5)
        elif compLevel == 4:
            d['weight'] = round(random.uniform(0.6, 0.7), 5)
        #d['weight'] = round(random.uniform(1, 1.5), 5)
        #labelsn[u] = d["weight"]
        d['estimate'] = 0
        d['time'] = 0
        nodel.append(u)
        # print(u, d["weight"], 1.0/d["weight"])
    for (u, v, d) in G.edges(data=True):
        if transLevel == 0:
            d['weight'] = round(random.uniform(1, 2), 5)
        elif transLevel == 1:
            d['weight'] = round(random.uniform(2, 3), 5)
        elif transLevel == 2:
            d['weight'] = round(random.uniform(3, 4), 5)
        elif transLevel == 3:
            d['weight'] = round(random.uniform(4, 5), 5)
        elif transLevel == 4:
            d['weight'] = round(random.uniform(5, 6), 5)
        labelse[(u, v)] = d["weight"]
        d['estimate'] = 0
        d['time'] = 0
        edgel.append((u, v))

    # add cloud
    G.add_node(cloud, weight=0.1, time=0, estimate=0)
    for n in G.nodes:
        if n != cloud:
            G.add_edge(n, cloud)
    for (u, v, d) in G.edges(data=True):
        if v == cloud:
            d['weight'] = random.uniform(20, 50)
            d['estimate'] = 0
            d['time'] = 0
    return G
def main1(G, iters):
    estimateArms(G)
    pos = random.choice(list(G.nodes))
    Pos = [pos]
    lps =[]
    for n in G.nodes:
        lps.append(n)
    sa = list(permutations(lps, service_num))
    context = contextData()
    context.update(pos=pos, demand=[random.choice([100, 100, 100]) for i in range(service_num)],
                   lastplace=random.choice(sa))
    # initialize the variables
    Pos = [(0,0), context.pos]
    Demand = [0, context.demand]
    lastplacement = context.last_placement
    Cost = []
    Cost_op = []
    avgC = []
    avgC_op = []
    regrets = []
    avgR = []

    # start learning
    start_time = time.time()
    for i in range(iters):
        placement = oracle_dp(G, context, i+1)
        cost, T, S, C = scCost(placement, G, context)
        #print(placement1, placement2)
        #print(cost1, cost2)
        #cost_op, T_op, S_op, C_op = scCost(placement_op, G, context)
        #print('cost:', cost2 )
        #print('cost:', cost, 'T:',T, 'S:', S, 'C:', C)
        Cost.append(cost)
        #Cost_op.append(cost_op)
        #regrets.append(cost1 - cost_op)
        #avgR.append(sum(regrets))
        avgC.append(sum(Cost)/len(Cost))
        #avgC_op.append(sum(Cost_op)/len(Cost_op))
        updateEstimations(placement, G, context)
        pos = move(pos, G)
        #print(pos)
        Pos.append(pos)
       # [random.choice([2.5, 55, 100]) for i in range(service_num)]
        context.update(pos, [2.5, 2.5, 100], placement)
        Demand.append(context.demand)



    x = [i for i in range(iters)]
    plt.plot(x, avgR)
    plt.xlabel("Learning slot")
    plt.ylabel("Regret")
    plt.savefig("regret3.eps", format="eps")
    plt.show()
    f = getShortestPathlist(G)
    c = getCapacitylist(G)
    p = [list(G.nodes).index(pos) for pos in Pos]
    lp = [list(G.nodes).index(i) for i in lastplacement]

    start_time = time.time()
    offline_op = offline(G, iters, Demand, p, f, c, lp)
    runtime2 = time.time() - start_time
    avgC_op = [offline_op / iters for i in range(iters)]

    # print(offline_op)
    # avgC_ol = offline_sc(G, iters, lastplacement, Pos, Demand)

    plt.plot(x, avgC, label='CMAB')
    plt.plot(x, avgC_op, label='opt')
    plt.xlabel("Learning slot")
    plt.ylabel("Time average cost")
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig("learnin9.eps", format="eps")
    plt.show()
def DP(G, context):
    #print(service_num)
    n = len(G.nodes)
    c = getCapacitylist(G)
    f = getShortestPathlist(G)
   # print(context.last_placement)
    lp = [list(G.nodes).index(p) for p in context.last_placement]
    d = context.demand
   # print(context.pos)
    pos =list(G.nodes).index(context.pos)
    Host = [[[] for i in range(n)] for i in range(service_num)]
    D = [[0.0 for i in range(n)] for i in range(service_num)]

    i = service_num - 1
    while( i >= 0 ):
        for j in range(n):
            minCost = inf
            if i == service_num-1:
                D[i][j] = d[i]*c[j] + switch_weight*f[lp[i]][j] + f[j][pos]
            elif i == 0:
                minCost = inf
                host = 0
                for k in range(n):
                    if cloud in list(G.nodes):
                        #print("cloud", list(G.nodes).index(cloud))
                        if (j not in Host[i+1][k] and j != k) or (j == list(G.nodes).index(cloud)):
                            cost = f[pos][j] + d[i]*c[j] + switch_weight*f[lp[i]][j] + f[j][k] + D[i+1][k]
                            if minCost >= cost:
                                minCost = cost
                                host = k
                    else:
                        #print("edge")
                        if (j not in Host[i+1][k] and j != k):
                            cost = f[pos][j] + d[i]*c[j] + switch_weight*f[lp[i]][j] + f[j][k] + D[i+1][k]
                            if minCost >= cost:
                                minCost = cost
                                host = k
                D[i][j] = minCost
                Host[i][j] = [host] + Host[i+1][host]
                # cost = f[pos][j] + d[i]*c[j] + f[lp[i]][j] + min(f[i][k]+D[i+1][k] for k in range(n))
            else:
                minCost = inf
                host = 0
                for k in range(n):
                    if cloud in list(G.nodes):
                        # print("cloud", list(G.nodes).index(cloud))
                        if (j not in Host[i + 1][k] and j != k) or (j == list(G.nodes).index(cloud)):
                            cost = d[i] * c[j] + switch_weight*f[lp[i]][j] + f[j][k] + D[i+1][k]
                            if minCost >= cost:
                                minCost = cost
                                host = k
                    else:
                        # print("edge")
                        if (j not in Host[i + 1][k] and j != k):
                            cost = d[i] * c[j] + switch_weight*f[lp[i]][j] + f[j][k] + D[i+1][k]
                            if minCost >= cost:
                                minCost = cost
                                host = k
                D[i][j] = minCost
                Host[i][j] = Host[i+1][host] + [host]
                # cost = d[i]*c[j] + min(f[i][k]+D[i+1][k] for k in range(n))
        i -= 1
    m = D[0].index(min(D[0]))

    arm = [list(G.nodes)[p] for p in [m] + Host[0][m]]
    # for a in Host[m]:
    #
    #
    # arm.append(list(G.nodes)[m])
    # for i in range(service_num-1):
    #     m = Host[i][m]
    #     arm.append(list(G.nodes)[m])
    return arm, min(D[0])
def DP_edge(G, context):
    # print("edge")
    # print("cloud:", list(G.nodes).index(cloud))
    n = len(G.nodes)
    c = getCapacitylist(G)
    f = getShortestPathlist(G)
   # print(context.last_placement)
    lp = [list(G.nodes).index(p) for p in context.last_placement]
    d = context.demand
   # print(context.pos)
    pos =list(G.nodes).index(context.pos)
    Host = [[[] for i in range(n-1)] for i in range(service_num)]
    D = [[0.0 for i in range(n-1)] for i in range(service_num)]

    i = service_num - 1
    while( i >= 0 ):
        for j in range(n):
            if j != list(G.nodes).index(cloud):
                # print(j)
                minCost = inf
                if i == service_num-1:
                    D[i][j] = d[i]*c[j] + switch_weight*f[lp[i]][j] + f[j][pos]
                elif i == 0:
                    minCost = inf
                    host = 0
                    for k in range(n):
                        if (k !=list(G.nodes).index(cloud)):
                            # print(k)
                            if (j not in Host[i+1][k] and j != k):
                                cost = f[pos][j] + d[i]*c[j] + switch_weight*f[lp[i]][j] + f[j][k] + D[i+1][k]
                                if minCost >= cost:
                                    minCost = cost
                                    host = k
                    D[i][j] = minCost
                    Host[i][j] = [host] + Host[i+1][host]
                    # cost = f[pos][j] + d[i]*c[j] + f[lp[i]][j] + min(f[i][k]+D[i+1][k] for k in range(n))
                else:
                    minCost = inf
                    host = 0
                    for k in range(n):
                        if (k !=list(G.nodes).index(cloud)):
                            # print(k)
                            if (j not in Host[i+1][k] and j != k):
                                cost = d[i]*c[j] + switch_weight*f[lp[i]][j] + f[j][k] + D[i+1][k]
                                if minCost >= cost:
                                    minCost = cost
                                    host = k
                    D[i][j] = minCost
                    Host[i][j] = Host[i+1][host] + [host]
                    # cost = d[i]*c[j] + min(f[i][k]+D[i+1][k] for k in range(n))
        i -= 1
    # print(Host)
    m = D[0].index(min(D[0]))
    # print([m] + Host[0][m])
    arm = [list(G.nodes)[p] for p in [m] + Host[0][m]]
    # for a in Host[m]:
    #
    #
    # arm.append(list(G.nodes)[m])
    # for i in range(service_num-1):
    #     m = Host[i][m]
    #     arm.append(list(G.nodes)[m])
    # print(arm)
    return arm, min(D[0])
def DynamicGreedy(iters, D, eps):
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
def main2(G, iters, th):
    # draw the network topology
    # edge = list(G.nodes)
    # edge.remove(cloud)
    # pos = nx.spring_layout(G, iterations=100)
    # nx.draw(G, pos, node_size=300)
    # nx.draw_networkx_labels(G, pos, labelsn)
    # nx.draw_networkx_edge_labels(G, pos, labelse)
    # plt.show()
    # labels = {}
    # labels[cloud] = r'$cloud$'
    # for e in edge:
    #     labels[e] = r'$e$'
    #
    # cloudedge = []
    # gridedge = []
    # for (u, v) in G.edges:
    #     if v == cloud:
    #         cloudedge.append((u, v))
    #     else:
    #         gridedge.append((u, v))
    # nx.draw_networkx_edges(G, pos,
    #                        edgelist=cloudedge,
    #                        width=10, alpha=0.5, edge_color='r')
    # nx.draw_networkx_edges(G, pos,
    #                        edgelist=gridedge,
    #                        width=1, alpha=0.5, edge_color='b')
    # nx.draw_networkx_labels(G, pos, labels, font_size=16)
    # #plt.savefig("network.eps", format='eps')
    # plt.show()
    # estimate Arms and print the inital network data
    estimateArms(G)
    Gc = G.copy()
    Ge = G.copy()
    # for (n, d) in G.nodes(data=True):
    #     print(d)
    # for (u, v ,d) in G.edges(data=True):
    #     print(d)

    # set up the context data
    lps =[]
    for n in G.nodes:
        if n != cloud:
            lps.append(n)
    sa = list(permutations(lps, service_num))
    pos = random.choice(lps)
    Pos = [pos]
    context = contextData()
    context.update(pos=pos, demand=[100, 100, 100],
                   lastplace=random.choice(sa))

    # initialize the variables
    Pos = [(0,0), context.pos]
    Demand = [0, context.demand]
    lastplacement = context.last_placement
    #print(lastplacement)
    CostCMAB = []
    Cost_op = []
    avgCMAB = []
    avgC_op = []
    regrets = []
    avgR = []

    # start learning
    start_time = time.time()
    for i in range(iters):
        # print("###########################################")
        # print("current position: ", context.pos)
        # print("current demand: ", context.demand)
        # print("last placement: ", context.last_placement)
        placement, cost = oracle_dp(G, context, i+1)
        cost, T, S, C = scCost(placement, G, context)
        # print("current placement and cost:", placement, cost)
        # print("###########################################")
        # print("\n\n\n")
        #print(placement1, placement2)
        #print(cost1, cost2)
        #cost_op, T_op, S_op, C_op = scCost(placement_op, G, context)
        #print('cost:', cost2 )
        #print('cost:', cost, 'T:',T, 'S:', S, 'C:', C)
        CostCMAB.append(cost)
        #Cost_op.append(cost_op)
        #regrets.append(cost1 - cost_op)
        #avgR.append(sum(regrets))
        avgCMAB.append(sum(CostCMAB)/len(CostCMAB))
        #avgC_op.append(sum(Cost_op)/len(Cost_op))
        updateEstimations(placement, G, context)
        pos = move(pos, G)
        #choices = [[2.5, 2.5, 2.5], [100, 100, 100], [2.5, 2.5, 2.5], [2.5, 100, 2.5], [100, 2.5, 2.5], [2.5, 2.5, 100]]
        # [random.choice([2.5, 25, 100]) for i in range(service_num)]
        context.update(pos, [random.choice([2.5, 2.5, 25, 25, 100]) for i in range(service_num)], placement)
        Pos.append(pos)
        Demand.append(context.demand)
    runtime1 = time.time() - start_time

    # SC
    CostCloud = []
    avgCloud = []
    context.update(Pos[1], Demand[1], lastplacement)
    for i in range(iters):
        placement = oracle_cloud()
        #print(placement, cost)
        cost, _, _, _ = scCost(placement, Gc, context)
        CostCloud.append(cost)
        avgCloud.append(sum(CostCloud)/len(CostCloud))
        updateEstimations(placement, Gc, context)
        if i != iters-1:
            context.update(Pos[i + 2], Demand[i + 2], placement)


    # SE
    CostEdge = []
    avgEdge = []
    context.update(Pos[1], Demand[1], lastplacement)
    for i in range(iters):

        placement = oracle_edge(Ge, context, i+1)
        #print(placement, cost)
        cost, _, _, _ = scCost(placement, Ge, context)
        CostEdge.append(cost)
        avgEdge.append(sum(CostEdge)/len(CostEdge))
        updateEstimations(placement, Ge, context)
        lastplacement = placement
        if i != iters-1:
            context.update(Pos[i + 2], Demand[i + 2], placement)

    # # heuristic
    # CostHeuristic = []
    # avgHeuristic = []
    # context.update(Pos[1], Demand[1], lastplacement)
    # for i in range(iters):
    #     placement = heuristic(G, context, i+1, th)
    #     # print(placement, cost)
    #     cost, _, _, _ = scCost(placement, G, context)
    #     CostHeuristic.append(cost)
    #     avgHeuristic.append(sum(CostHeuristic) / len(CostHeuristic))
    #     updateEstimations(placement, G, context)
    #     lastplacement = placement
    #     context.update(Pos[i + 1], Demand[i + 1], placement)

    #print(Demand)
    #plot and save the image
    f = getShortestPathlist(G)
    c = getCapacitylist(G)
    p = [list(G.nodes).index(pos) for pos in Pos]
    lp = [list(G.nodes).index(i) for i in lastplacement]
    # offline_op = offline(G, iters, Demand, p, f, c, lp)
    # b = bound(iters, G, service_num)
    # print(1+b/offline_op)
    offline_op = avgCMAB[-1]*iters
    avgC_op = [offline_op / iters for i in range(iters)]

    # for (n, d) in G.nodes(data=True):
    #     print(d)
    # for (u, v ,d) in G.edges(data=True):
    #     print(d)
    x = [i for i in range(iters)]
    #plt.plot(x, avgCMAB, label='CMAB')
    # # plt.plot(x, avgC_op, label='opt')
    # plt.plot(x, avgCloud, label='cloud')
    # plt.plot(x, avgEdge, label='edge')
    # #plt.plot(x, avgHeuristic, label='heuristic')
    # plt.xlabel("Learning slot")
    # plt.ylabel("Time average cost")
    # plt.ylim(bottom=0)
    # plt.legend()
    # plt.savefig("learnin9.eps", format="eps")
    # plt.show()
    return avgCMAB[-1], avgCloud[-1]+10.0, avgEdge[-1], avgC_op[-1]
def mainLearning(G, iters, eps):
    # global service_num
    # service_num = 3
    estimateArms(G)
    pos = random.choice(list(G.nodes))
    context = contextData()
    context.update(pos=pos, demand=[random.choice([2.5, 25, 100]) for i in range(service_num)],
                   lastplace=random.sample(list(G.nodes), service_num))

    Pos = [(0, 0), context.pos]
    Demand = [0, context.demand]
    lastplacement = context.last_placement
    Cost = []
    EstCost = []
    avgE = []
    avgC = []
    Regret = []
    AvgRegret = []
    r = 0.0
    ar = 0.0
    #Context = [context]
    # start online learning
    start_time = time.time()
    for i in range(iters):
        #(n, d) = list(G.nodes(data=True))[10]
        #print(d['estimate'], d['weight'])
        op_placement, op_cost = oracle_optimal(G, context)
        #cost1, T, S, C = scCost(op_placement, G, context)
        #print(op_cost, cost1)
        placement1, estcost = oracle_dp(G, context, i + 1)
        cost1, T, S, C = scCost(placement1, G, context)
        #print(cost1, estcost)
        r = r + abs(cost1-estcost)
        Regret.append(r)
        Cost.append(cost1)
        EstCost.append(estcost)
        if abs(sum(Cost) / len(Cost) - sum(EstCost) / len(EstCost)) > 3.0:
            ar = ar + abs(sum(Cost) / len(Cost) - sum(EstCost) / len(EstCost))-3
        #print(abs(sum(Cost) / len(Cost) - sum(EstCost) / len(EstCost)))
        #02.print(ar)
        AvgRegret.append(ar)
        avgC.append(sum(Cost) / len(Cost))
        avgE.append(sum(EstCost) / len(EstCost))
        updateEstimations(placement1, G, context)
        pos = move(pos, G)
        context.update(pos, [random.choice([2.5, 2.5, 25, 25, 100]) for i in range(service_num)], placement1)
        #Context.append(context)
        Pos.append(pos)
        Demand.append(context.demand)
    runtime = time.time() - start_time
    #open("regretc%fi%i"%(explore_ratio, iters), "w").write(str(Regret))
    #open("avgregretc%fi%i"%(explore_ratio, iters), "w").write(str(AvgRegret))

    # # greedy
    # actions = [list(a) for a in (permutations(list(G.nodes), service_num))]
    # values = [[] for a in actions]
    # Costeg = []
    # context.update(pos=Pos[1], demand=Demand[1], lastplace=lastplacement)
    # for i in range(iters):
    #     p = np.random.random()
    #     if not any(values) or p < eps:
    #         plc = random.sample(list(G.nodes), service_num)
    #     else:
    #         m = []
    #         for v in values:
    #             if v != []:
    #                 m.append(sum(v) / len(v))
    #             else:
    #                 m.append(1000.0)
    #         plc = actions[np.argmin(m)]
    #     cost, _, _, _ = scCost(plc, G, context)
    #     if i != iters-2:
    #         context.update(Pos[i+2], Demand[i+2], plc)
    #     values[actions.index(plc)].append(cost)
    #     Costeg.append(cost)
    # #print(Costeg)
    # # adaptive greedy
    # actions = [list(a) for a in (permutations(list(G.nodes), service_num))]
    # values = [[] for a in actions]
    # Costag = []
    # context.update(pos=Pos[1], demand=Demand[1], lastplace=lastplacement)
    # for i in range(iters):
    #     e = eps * (1.0 - i / iters)
    #     p = np.random.random()
    #     #print(p, eps)
    #     if (not any(values)) or p < e:
    #         #print("1")
    #         plc = random.sample(list(G.nodes), service_num)
    #     else:
    #         #print("0")
    #         m = []
    #         for v in values:
    #             if v != []:
    #                 m.append(sum(v) / len(v))
    #             else:
    #                 m.append(1000.0)
    #         plc = actions[np.argmin(m)]
    #     cost, _, _, _ = scCost(plc, G, context)
    #     if i != iters-2:
    #         context.update(Pos[i + 2], Demand[i + 2], plc)
    #     values[actions.index(plc)].append(cost)
    #     Costag.append(cost)
    #     #print(plc)
    # Avgeg = [sum(Costeg[0:i + 1]) / len(Costeg[0:i + 1]) for i in range(len(Costeg))]
    # Avgag = [sum(Costag[0:i + 1]) / len(Costag[0:i + 1]) for i in range(len(Costag))]
    #
    #
    # offline
    f = getShortestPathlist(G)
    c = getCapacitylist(G)
    p = [list(G.nodes).index(pos) for pos in Pos]
    lp = [list(G.nodes).index(i) for i in lastplacement]
    offline_op = 40.0*iters#offline(G, iters, Demand, p, f, c, lp)
    avgC_op = [offline_op / iters for i in range(iters)]

    # open("cmabn%is%i"%(len(list(G.nodes)), service_num), "w").write(str(Cost))
    # open("egreedyn%is%i" % (len(list(G.nodes)), service_num), "w").write(str(Costeg))
    # open("agreedyn%is%i" % (len(list(G.nodes)), service_num), "w").write(str(Costag))
    x = [i for i in range(iters)]
    plt.plot(x, avgC, '-.', label='CMAB')
    plt.plot(x, avgC_op, label='Offline optimum')
    #open("cmab1c%fi%i"%(explore_ratio, iters), "w").write(str(Cost))
    #open("offline1", "w").write(str(offline_op / iters))
    #plt.plot(x, Avgeg, '--', label='$\epsilon$-greedy')
    #plt.plot(x, Avgag, ':', label='adaptive greedy')
    plt.xlabel("Learning slot")
    plt.ylabel("Time average cost")
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig("learning13.eps", format="eps")
    plt.savefig("learning13.png", format="png")
    plt.show()
    return runtime
def replot():
    AvgRegret = ast.literal_eval(open("avgregretc0.300000i10000", 'r').readline())
    Cost = ast.literal_eval(open("cmab1c0.300000i10000", 'r').readline())
    Offline = ast.literal_eval(open('offline', 'r').readline())
    x = [i for i in range(len(Cost))]
    avgc = [sum(Cost[0:i + 1]) / len(Cost[0:i + 1]) for i in range(len(Cost))]
    navgc = [a/(max(avgc)-Offline) for a in avgc]
    avgo = [Offline/(max(avgc)-Offline) for i in x]
    plt.plot(x[0:2000], navgc[0:2000], '-.', label='BandEdge', zorder=3)
    plt.plot(x[0:2000], avgo[0:2000], label='Offline optimum', zorder=3)
    plt.xlabel("Learning slot")
    plt.ylabel("Time average cost")
    plt.ylim(bottom=0)
    plt.grid(linestyle='--', zorder=0)
    plt.legend()
    plt.savefig("replotlearningnormalized.eps", format="eps")
    plt.savefig("replotlearningnormalized.png", format="png")
    plt.show()
def replot2y():
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Learning slots")
    ax1.set_ylabel("Time average cost")
    AvgRegret = ast.literal_eval(open("avgregretc0.300000i10000", 'r').readline())
    Cost = ast.literal_eval(open("cmab1c0.300000i10000", 'r').readline())
    Offline = ast.literal_eval(open('offline', 'r').readline())
    x = [i for i in range(len(Cost))]
    avgc = [sum(Cost[0:i + 1]) / len(Cost[0:i + 1]) for i in range(len(Cost))]
    navgc = [a/(max(avgc)-Offline) for a in avgc]
    avgo = [Offline/(max(avgc)-Offline) for i in x]
    l1, = ax1.plot(x[0:4000], navgc[0:4000], '--', label='BandEdge', zorder=3)
    l2, = ax1.plot(x[0:4000], avgo[0:4000], label='Offline optimum', zorder=3)
    # plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    ax2 = ax1.twinx()
    ax2.set_ylabel("regret")
    l3, = ax2.plot(x[0:4000], AvgRegret[0:4000], ':', label="regret", zorder=3)
    fig.tight_layout()
    ax1.grid(linestyle='--', zorder=0)
    #ax2.grid(linestyle='--', zorder=0)
    plt.ylim(bottom=0)
    plt.legend([l1, l2, l3], ["BandEdge", "Offline optimum", "Regret"], loc='right')
    plt.savefig("replotlearningnormalized1.eps", format="eps")
    plt.savefig("replotlearningnormalized1.png", format="png")
    plt.show()
def replotRuntime():
    Marker = ["o", "v", "^", "x"]
    for n in [3, 4, 5, 6]:
        RT = ast.literal_eval(open("runtimeN%i"%n, 'r').readline())
        plt.plot([i+3 for i in range(len(RT))], RT, label="%i nodes"%(n*n), marker=Marker[n-3], zorder=3)
    plt.xlabel("Number of Services")
    plt.ylabel("Running Time (s)")
    plt.grid(linestyle='--', zorder=0)
    plt.legend()
    plt.savefig("../paper/figs/replotruntime.eps", format="eps")
    plt.savefig("replotruntime.png", format="png")
    plt.show()

def avgList(list):
    return [sum(list[0:i + 1]) / len(list[0:i + 1]) for i in range(len(list))]
def replotMiniWifiLearning():
    # Cost1 = [float(r) for r in open("gCostg.txt", 'r').readlines()]
    # Cost2 = [float(r) for r in open("Cost1000.txt", 'r').readlines()]
    #gCost = [float(r) for r in open("gCost1000.txt", 'r').readlines()]
    Cost = [float(r) for r in open("Costg3 - Copy.txt", 'r').readlines()]
    #Cost5 = [float(r) for r in open("tCost1000.txt", 'r').readlines()]
    gCost =  [float(r) for r in open("gCostg.txt", 'r').readlines()]
    # gCost2 = [float(r) for r in open("gCost1000.txt", 'r').readlines()]
    # gCost3 = [float(r) for r in open("gCost1000.txt", 'r').readlines()]
    # gCost4 = [float(r) for r in open("tCost1000.txt", 'r').readlines()]





    # gCost = [gCost1, gCost2, gCost3, gCost4]


    Marker = ["o", "v", "^", "<", ">", "+", "x", "|", "_"]
    Y1 = []
    Y2 = []
    X = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    AvgCost = []
    AvggCost = []
    for n in X:
        AvgCost.append(avgList(Cost)[n-1])
        AvggCost.append(avgList(gCost)[n-1])
    # for c in gCost:
    #     AvggCost = []
    #     for n in X:
    #         AvggCost.append(avgList(c)[n-1])
    #     Y2.append(AvggCost)


    plt.plot(X, AvgCost, label="BandEdge", marker=random.choice(Marker), zorder=3)
    plt.plot(X, AvggCost, label="$\epsilon$-greedy", marker=random.choice(Marker), zorder=3)
    plt.plot(X, [float(open("offline100_1.txt").read()) for i in range(len(X))], label="offline optimum", zorder=3)
    plt.xlabel("Learning slots")
    plt.ylabel("Average response time")
    plt.ylim(bottom=0)
    plt.grid(linestyle='--', zorder=0)
    plt.legend()
    plt.savefig("MiniwifiLearning1.eps", format="eps")
    plt.savefig("MiniwifiLearning1.png", format="png")
    plt.show()

def replotregret():
    AvgRegret = ast.literal_eval(open("avgregretc0.300000i10000", 'r').readline())
    #Offline = ast.literal_eval(open('offline', 'r').readline())
    x = [i for i in range(len(AvgRegret))]
    plt.plot(x, AvgRegret, zorder=3)
    plt.xlabel("Learning slot")
    plt.ylabel("Regret")
    plt.ylim(bottom=0, top=25000)
    plt.grid(linestyle='--', zorder=0)
    plt.savefig("replotsingleregret.eps", format="eps")
    plt.savefig("replotsingleregret.png", format="png")
    plt.show()
def plotRegret(iters):
    #Regret = ast.literal_eval(open('regretc0.000000i1000', 'r').readline())
    erlist = [0.05,  0.1,  0.3,  0.5,  1.0]

    # x = [i+1 for i in range(iters)]
    #plt.plot(x, Regret, zorder=3)
    for er in erlist:
        AvgRegret = ast.literal_eval(open("avgregretc%fi%i" % (er, iters), 'r').readline())
        plt.plot([i+1 for i in range(len(AvgRegret))], AvgRegret, zorder=3, label="c = %f"%er)
    plt.xlabel("Learning slots")
    plt.ylabel("Regret")
    plt.grid(linestyle='--', zorder=0)
    plt.legend()
    plt.savefig("replotregret1.eps", format="eps")
    plt.savefig("replotregret1.png", format="png")
    plt.show()
def plotRatio(iters):
    #Regret = ast.literal_eval(open('regretc0.000000i1000', 'r').readline())
    erlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    x = [i+1 for i in range(iters)]
    #plt.plot(x, Regret, zorder=3)
    for er in erlist:
        Cost = ast.literal_eval(open("cmab1c%fi%i" % (er, iters), 'r').readline())
        AvgCost = [sum(Cost[0:i + 1]) / len(Cost[0:i + 1]) for i in range(len(Cost))]
        plt.plot(x, AvgCost, zorder=3, label="c = %f"%er)
    plt.xlabel("Learning slots")
    plt.ylabel("Time average cost")
    plt.grid(linestyle='--', zorder=0)
    plt.legend()
    plt.savefig("replotratio1.eps", format="eps")
    plt.savefig("replotratio1.png", format="png")
    plt.show()
def plotExploreratio(G, iters):
    estimateArms(G)
    H1 = G.copy()
    H2 = G.copy()
    H3 = G.copy()
    H4 = G.copy()
    H5 = G.copy()
    Hs = [H1, H2, H3, H4, H5]
    # initialize the variables
    lps = []
    for n in G.nodes:
        if n != cloud:
            lps.append(n)
    sa = list(permutations(lps, service_num))
    pos = random.choice(lps)
    Pos = [pos]
    context = contextData()
    context.update(pos=pos, demand=[2.5, 25, 100],
                   lastplace=random.choice(sa))
    Pos = [(0, 0), context.pos]
    Demand = [0, context.demand]
    lastplacement = context.last_placement

    R = [0.1, 0.3, 0.5, 0.7, 0.9]
    Rcost = []
    for i, r in enumerate(R):
        rcost = []
        ecost = []
        avgrcost = []
        avgecost =[]
        global explore_ratio
        explore_ratio = r
        H = Hs[i]
        for j in range(iters):
            placement, eost = oracle_dp(H, context, i + 1)
            cost, T, S, C = scCost(placement, H, context)
            rcost.append(cost)
            ecost.append(eost)
            avgrcost.append(sum(rcost) / len(rcost))
            avgecost.append(sum(ecost)/len(ecost))
            updateEstimations(placement, H, context)
            if r == R[0]:
                pos = move(pos, H)
                dem = [random.choice([2.5, 2.5, 25, 25, 100]) for i in range(service_num)]
                Pos.append(pos)
                Demand.append(context.demand)
            else:
                if j != iters - 2:
                    pos = Pos[i+2]
                    dem = Demand[i+2]
            context.update(pos, dem, placement)
        context.update(Pos[1], Demand[1], lastplacement)
        Rcost.append(avgrcost)
    x = [i for i in range(iters)]
    for i, r in enumerate(R):
        open("avgrcostc%fi%i" % (explore_ratio, iters), "w").write(str(avgecost))
        open("avgecostc%fi%i" % (explore_ratio, iters), "w").write(str(avgecost))
        plt.plot(x, Rcost[i], label='c=%f'%r)
    plt.xlabel("Learning slot")
    plt.ylabel("Time average cost")
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig("exploreratio1.eps", format="eps")
    plt.show()
    # plt.plot(x, avgCMAB, label='CMAB')
    # # plt.plot(x, avgC_op, label='opt')
    # plt.plot(x, avgCloud, label='cloud')
    # plt.plot(x, avgEdge, label='edge')
    # #plt.plot(x, avgHeuristic, label='heuristic')
    # plt.xlabel("Learning slot")
    # plt.ylabel("Time average cost")
    # plt.ylim(bottom=0)
    # plt.legend()
    # plt.savefig("learnin9.eps", format="eps")
    # plt.show()
def plotServiceNumRuntime(i, s, n):

    Run1 = []
    Run2 = []
    for sn in range(3, s+1):
        G = createNetwork(5, transLevel=0, compLevel=0)
        global service_num
        service_num = sn
        #run1, run2, __, __ = main1(G, i, 5)
        run1 = mainLearning(G, i)
        Run1.append(run1)

    for sn in range(3, s+1):
        G = createNetwork(sn, transLevel=0, compLevel=0)
        service_num = 3
        run2 = mainLearning(G, i)
        Run2.append(run2)
    x = [i for i in range(3, s+1)]
    plt.plot(x, Run1, label='service', marker='o')
    plt.plot(x, Run2, label='node size', marker='v')
    plt.xlabel("# of services/node size")
    plt.xticks(x, x)
    plt.ylabel("Runtime (s)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig("runtime2.eps", format="eps")
    plt.show()
def plotTransWeight(n):
    CMAB = []
    Cloud = []
    Edge = []
    # Offline = []
    # heuristic = []
    # for i in range(5):
    #     G = createNetwork(5, i, 3)
    #     cmab, cloud, edge, offline = main2(G, n, None)
    #     CMAB.append(cmab)
    #     Cloud.append(cloud)
    #     Edge.append(edge)
    #     Offline.append(offline)
    # open("cmabtw", 'w').write(str(CMAB))
    # open("sctw", 'w').write(str(Cloud))
    # open("setw", 'w').write(str(Edge))
    CMAB = ast.literal_eval(open("cmabtw", 'r').readline())
    Cloud = ast.literal_eval(open("sctw", 'r').readline())
    Edge = ast.literal_eval(open("setw", 'r').readline())
    Y = CMAB + Cloud + Edge
    N = (max(Y)+min(Y))/2.0
    CMAB = [y / N for y in CMAB]
    Cloud = [y / N for y in Cloud]
    Edge = [y / N for y in Edge]
    ILP = [1.0 for i in range(5)]
    labels = [1,2,3,4,5]
    x = np.arange(len(labels))
    width = 0.3
    plt.bar(x - 3 * width / 2, CMAB, width, label='BandEdge', hatch='//', color='#4DBEEE', zorder=3)
    plt.bar(x - width / 2, Cloud, width, label='SC', hatch='..', color='#A2142F', zorder=3)
    plt.bar(x + width / 2, Edge, width, label='SE', hatch='++', color='#EDB120', zorder=3)
    plt.grid(linestyle='--', zorder=0)
    plt.ylabel('Performance ratio', fontsize=15)
    plt.xlabel('Transmission delay levels on edge', fontsize = 15)
    plt.xticks(x, labels)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    plt.savefig("../paper/figs/transmissionLevel2.eps", format='eps')
    plt.show()
def plotComputingWeight(n):
    CMAB = []
    Cloud = []
    Edge = []
    # Offline = []
    # heuristic = []
    # for i in range(5):
    #     G = createNetwork(5, 3, i)
    #     cmab, cloud, edge, offline = main2(G, n, None)
    #     CMAB.append(cmab)
    #     Cloud.append(cloud)
    #     Edge.append(edge)
    #     Offline.append(offline)
    # open("cmabcw", 'w').write(str(CMAB))
    # open("sccw", 'w').write(str(Cloud))
    # open("secw", 'w').write(str(Edge))
    CMAB = ast.literal_eval(open("cmabcw", 'r').readline())
    Cloud = ast.literal_eval(open("sccw", 'r').readline())
    Edge = ast.literal_eval(open("secw", 'r').readline())
    # CMAB = [Cost1[i] / Cost2[i] for i in range(5)]
    # Cloud = [Cost3[i] / Cost2[i] for i in range(5)]
    # Edge = [Cost4[i] / Cost2[i] for i in range(5)]
    # ILP = [1.0 for i in range(5)]
    Y = CMAB + Cloud + Edge
    N = (max(Y)+min(Y))/2.0
    CMAB = [y / N for y in CMAB]
    Cloud = [y / N for y in Cloud]
    Edge = [y / N for y in Edge]
    labels = [1, 2, 3, 4, 5]
    x = np.arange(len(labels))
    width = 0.3
    plt.bar(x - 3 * width / 2, CMAB, width, label='BandEdge', hatch='//', color='#4DBEEE', zorder=3)
    plt.bar(x - width / 2, Cloud, width, label='SC', hatch='..', color='#A2142F', zorder=3)
    plt.bar(x + width / 2, Edge, width, label='SE', hatch='++', color='#EDB120', zorder=3)
    plt.grid(linestyle='--', zorder=0)
    plt.ylabel('Performance ratio', fontsize=15)
    plt.xlabel('Computing delay levels on edge', fontsize=15)
    plt.xticks(x, labels)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    plt.savefig("../paper/figs/computationLevel2.eps", format='eps')
    plt.show()
    #plt.close()
def plotLearningSlot(n, t, c, th):
    CMAB = []
    Cloud = []
    Edge = []
    Offline = []
    heuristic = []
    for i in range(100, 501, 100):
        G = createNetwork(n, t, c)
        cmab, cloud, edge, offline = main2(G, i, None)
        CMAB.append(cmab)
        Cloud.append(cloud)
        Edge.append(edge)
        Offline.append(offline)
    open("cmabls", 'w').write(str(CMAB))
    open("scls", 'w').write(str(Cloud))
    open("sels", 'w').write(str(Edge))
    # CMAB = ast.literal_eval(open("cmabls", 'r').readline())
    # Cloud = ast.literal_eval(open("scls", 'r').readline())
    # Edge = ast.literal_eval(open("sels", 'r').readline())
    Y = CMAB + Cloud + Edge
    N = (max(Y)+min(Y))/2.0
    CMAB = [y/N for y in CMAB]
    Cloud = [y/N for y in Cloud]
    Edge = [y/N for y in Edge]
    labels = [100, 200, 300, 400, 500]
    x = np.arange(len(labels))
    width = 0.3
    plt.bar(x - 3 * width / 2, CMAB, width, label='BandEdge', hatch='//', color='#4DBEEE',zorder=3)
    plt.bar(x - width / 2, Cloud, width, label='SC', hatch='..', color='#A2142F', zorder=3)
    plt.bar(x + width / 2, Edge, width, label='SE', hatch='++', color='#EDB120', zorder=3)
    plt.grid(linestyle='--', zorder=0)
    plt.ylabel('Performance ratio')
    plt.xlabel("Learning slots")
    plt.xticks(x, labels)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    plt.savefig("learningslot2.eps", format="eps")
    plt.show()
def plotSwitchCost(i):
    CMAB = []
    Cloud = []
    Edge = []
    Offline = []
    heuristic = []
    # for sw in labels:
    #     global switch_weight
    #     switch_weight = sw
    #     G = createNetwork(5, 3, 3)
    #     cmab, cloud, edge, offline = main2(G, i, 80)
    #     CMAB.append(cmab)
    #     Cloud.append(cloud)
    #     Edge.append(edge)
    #     Offline.append(offline)
    # open("cmabsw", 'w').write(str(CMAB))
    # open("scsw", 'w').write(str(Cloud))
    # open("sesw", 'w').write(str(Edge))
    CMAB = ast.literal_eval(open("cmabsw", 'r').readline())
    Cloud = ast.literal_eval(open("scsw", 'r').readline())
    Edge = ast.literal_eval(open("sesw", 'r').readline())
    Y = CMAB + Cloud + Edge
    N = (max(Y)+min(Y))/2.0
    CMAB = [y / N for y in CMAB]
    Cloud = [y / N for y in Cloud]
    Edge = [y / N for y in Edge]
    labels = [1.0, 2.0, 3.0, 4.0, 5.0]
    # CMAB = [Cost1[i] / Cost2[i] for i in range(5)]
    # Cloud = [Cost3[i] / Cost2[i] for i in range(5)]
    # Edge = [Cost4[i] / Cost2[i] for i in range(5)]
    # CMAB = Cost1
    # Cloud = Cost2
    # Edge = Cost4
    # ILP = [1.0 for i in range(5)]
    x = np.arange(len(labels))
    width = 0.3
    plt.bar(x - 3 * width / 2, CMAB, width, label='BandEdge', hatch='//', color='#4DBEEE', zorder=3)
    plt.bar(x - width / 2, Cloud, width, label='SC', hatch='..', color='#A2142F', zorder=3)
    plt.bar(x + width / 2, Edge, width, label='SE', hatch='++', color='#EDB120', zorder=3)
    #plt.bar(x + 3 * width / 2, heuristic, width, label='Heuristic', hatch='+')
    plt.ylabel('Performance ratio',fontsize=15) #,fontsize=15
    plt.xlabel("The weight of switching cost",fontsize=15)
    plt.xticks(x, labels)
    plt.grid(linestyle='--', zorder=0)
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    plt.savefig("../paper/figs/switchingcost2.eps", format="eps")
    plt.show()
def bound(n, G, s):
    m = len(list(G.nodes))
    a = sqrt(m-1)
    context = contextData()
    context.update(pos=random.choice(list(G.nodes)), demand=[random.choice([2.5, 25, 100]) for i in range(service_num)],
                   lastplace=random.sample(list(G.nodes), service_num))
    S = permutations(list(G.nodes), s)

    SC = [scCost(s, G, context)[0] for s in S]
    SC.sort()
    min = SC[1]-SC[0]
    max = SC[-1]-SC[0]
    maxLength = 4.0*a + s*1.0
    print(min, maxLength, max, s, a, m ,n)
    return (6*log(n)/(min/maxLength)**2 + pi**2/3 + 1)*m*max/n
def heuristic(G, context, i, threshhold):
    placement = oracle_edge(G, context, i)
    c1, _, _, _ = scCost(placement, G, context)
    c2, _, _, _ = scCost([cloud for i in range(service_num)], G, context)
    # if c1 > c2:
    #     placement = [cloud for i in range(service_num)]
    if c1 > threshhold:
        placement = [cloud for i in range(service_num)]
    print(placement)
    return placement
def mobilitybar():
    RWCost = [float(r) for r in open("Costrw.txt", 'r').readlines()]
    RWeg = [float(r) for r in open("egCostrw.txt", 'r').readlines()]
    RWdyg = [float(r) for r in open("dygCostrw.txt", 'r').readlines()]
    RDCost = [float(r) for r in open("Costrd.txt", 'r').readlines()]
    RDeg = [float(r) for r in open("egCostrd.txt", 'r').readlines()]
    RDdyg = [float(r) for r in open("dygCostrd.txt", 'r').readlines()]
    TVCCost = [float(r) for r in open("CostTVC.txt", 'r').readlines()]
    TVCeg = [float(r) for r in open("dygCostTVC.txt", 'r').readlines()]
    TVCdyg = [float(r) for r in open("egCostTVC.txt", 'r').readlines()]
    GMCost = [float(r) for r in open("CostGM.txt", 'r').readlines()]
    GMeg = [float(r) for r in open("egCostGM.txt", 'r').readlines()]
    GMdyg = [float(r) for r in open("dygCostGM.txt", 'r').readlines()]
    RPCost = [float(r) for r in open("CostRP.txt", 'r').readlines()]
    RPeg = [float(r) for r in open("egCostRP.txt", 'r').readlines()]
    RPdyg = [float(r) for r in open("dygCostRP.txt", 'r').readlines()]
    ILP = [1.0 for i in range(5)]
    labels = ("RW", "RD", "TVC", "GM", "RP")
    x = np.arange(len(labels))
    y1 = [sum(RWeg)/len(RWeg), sum(RDeg)/len(RDeg), sum(TVCeg)/len(TVCeg), sum(GMeg)/len(GMeg), sum(RPeg)/len(RPeg)]
    e1 = [mean_confidence_interval(RWeg), mean_confidence_interval(RDeg), mean_confidence_interval(TVCeg),
          mean_confidence_interval(GMeg), mean_confidence_interval(RPeg)]
    y2 = [sum(RWdyg) / len(RWdyg), sum(RDdyg) / len(RDdyg), sum(TVCdyg) / len(TVCdyg), sum(GMdyg) / len(GMdyg),
          sum(RPdyg) / len(RPdyg)]
    e2 = [mean_confidence_interval(RWdyg), mean_confidence_interval(RDdyg), mean_confidence_interval(TVCdyg),
          mean_confidence_interval(GMdyg), mean_confidence_interval(RPdyg)]
    y3 = [sum(RWCost) / len(RWCost), sum(RDCost) / len(RDCost), sum(TVCCost) / len(TVCCost), sum(GMCost) / len(GMCost),
          sum(RPCost) / len(RPCost)]
    e3= [mean_confidence_interval(RWCost), mean_confidence_interval(RDCost), mean_confidence_interval(TVCCost),
          mean_confidence_interval(GMCost), mean_confidence_interval(RPCost)]
    # EstY = []
    # CostY = [10.1781151891, 11.3251267552, 11.4250441456, 13.2298670268]
    # width = 0.5
    width = 0.3
    # fig, ax = plt.subplots()
    # formatter = FuncFormatter(millions)
    # ax.yaxis.set_major_formatter(formatter)

    plt.bar(x - width, y3, width, yerr=e1, capsize=7, label="BandEdge", zorder=3, hatch='//', color='#4DBEEE')
    plt.bar(x, y2, width, yerr=e2, capsize=7, label="adaptive greedy", zorder=3, hatch='..', color='#A2142F')
    plt.bar(x + width, y1, width, yerr=e3, capsize=7, label="$\epsilon$-greedy", zorder=3, hatch='++', color='#EDB120')

    # plt.bar(x - width / 2, y1, width, yerr=e1, capsize=7, label="RW")
    # plt.bar(x + width / 2, y2, width, yerr=e2, capsize=7, label="RD")
    # plt.bar(x - 3 * width / 2, CMAB, width, label='CMAB')
    # plt.bar(x - width / 2, Cloud, width, label='Cloud', hatch='-')
    # plt.bar(x + width / 2, Edge, width, label='Edge', hatch='/')
    # plt.bar(x + 3 * width / 2, heuristic, width, label='Heuristic')
    plt.ylabel('Average response time (s)', fontsize=15)
    plt.xlabel('Mobility models', fontsize=15)
    plt.xticks(x, labels)
    plt.grid(linestyle='--', zorder=0)
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    plt.savefig("mobilitymodel2.eps", format='eps')
    plt.savefig("mobilitymodel2.png", format='png')
    plt.show()
def delaybar():
    AvgCost = [float(r) for r in open("avgrestime3.txt", 'r').readlines()]
    Cost = []
    for i, a in enumerate(AvgCost):
        if i == 0:
            Cost.append(a)
        else:
            Cost.append((i + 1) * AvgCost[i] - i * AvgCost[i - 1])
    #x = [i for i in range(len(AvgCost))]
    EstCost = [float(r) for r in open("estimatetime3.txt", 'r').readlines()]
    AvgEst = [sum(EstCost[0:i + 1]) / len(EstCost[0:i + 1]) for i in range(len(EstCost))]

    labels = ("20", "60", "100")
    x = np.arange(len(labels))
    # Y, E = readdata()
    # y1 = [i for i in Y[0::2]]
    # e1 = [i for i in E[0::2]]
    # y2 = [i for i in Y[1::2]]
    # e2 = [i for i in E[1::2]]
    # y1 = []
    # y1 = [14.2922887325, 13.0282305717, 10.7465812772]
    # e1 = [3.34486096597, 1.40416287132, 1.4395536627]
    # y2 = [11.2347782986, 11.2534561094, 10.4709706704]
    # e2 = [2.3941068874, 1.06979879419, 1.02433588339]
    y1 = [AvgCost[20], AvgCost[60], AvgCost[99]]
    e1 = [mean_confidence_interval(Cost[0:20]), mean_confidence_interval(Cost[0:60]), mean_confidence_interval(Cost)]
    y2 = [AvgEst[20], AvgEst[60], AvgEst[99]]
    e2 = [mean_confidence_interval(EstCost[0:20]), mean_confidence_interval(EstCost[0:60]), mean_confidence_interval(EstCost)]

    width = 0.4

    # fig, ax = plt.subplots()
    # formatter = FuncFormatter(millions)
    # ax.yaxis.set_major_formatter(formatter)
    plt.bar(x-width/2, y1, width, yerr=e1, capsize=7, label="Mininet Experiment")
    plt.bar(x+width/2, y2, width, yerr=e2, capsize=7, label="Model Estimation")
    # plt.bar(x - 3 * width / 2, CMAB, width, label='CMAB')
    # plt.bar(x - width / 2, Cloud, width, label='Cloud', hatch='-')
    # plt.bar(x + width / 2, Edge, width, label='Edge', hatch='/')
    # plt.bar(x + 3 * width / 2, heuristic, width, label='Heuristic')
    plt.ylabel('Average response time (s)')
    plt.xlabel('Learning slot')
    plt.xticks(x, labels)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(linestyle='--', zorder=0)
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    plt.savefig("responsevsestimationbar.eps", format='eps')
    plt.savefig("responsevsestimationbar.png", format='png')
    plt.show()
def linkdelaybar():
    d1Cost = [float(r) for r in open("Costtd1.txt", 'r').readlines()]
    d1dyg = [float(r) for r in open("dygCosttd1.txt", 'r').readlines()]
    d1eg = [float(r) for r in open("egCosttd1.txt", 'r').readlines()]
    d2Cost = [float(r) for r in open("Costtd2.txt", 'r').readlines()]
    d2dyg = [float(r) for r in open("dygCosttd2.txt", 'r').readlines()]
    d2eg = [float(r) for r in open("egCosttd2.txt", 'r').readlines()]
    d3Cost = [float(r) for r in open("Costtd3.txt", 'r').readlines()]
    d3dyg = [float(r) for r in open("dygCosttd3.txt", 'r').readlines()]
    d3eg = [float(r) for r in open("egCosttd3.txt", 'r').readlines()]

    Y1 = [d1Cost, d2Cost, d3Cost]
    Y2 = [d1dyg, d2dyg, d3dyg]
    Y3 = [d1eg, d2eg, d3eg]
    labels = ("50-100ms", "150-200ms", "250-300ms")
    x = np.arange(len(labels))
    y1 = [sum(y) / len(y) for y in Y1]
    e1 = [mean_confidence_interval(y) for y in Y1]
    y2 = [sum(y) / len(y) for y in Y2]
    e2 = [mean_confidence_interval(y) for y in Y2]
    y3 = [sum(y) / len(y) for y in Y3]
    e3 = [mean_confidence_interval(y) for y in Y3]

    width = 0.3
    plt.bar(x - width, y1, width, yerr=e1, capsize=7, label="BandEdge", zorder=3, hatch='//', color='#4DBEEE')
    plt.bar(x, y2, width, yerr=e2, capsize=7, label="adaptive greedy", zorder=3, hatch='..', color='#A2142F')
    plt.bar(x + width, y3, width, yerr=e3, capsize=7, label="$\epsilon$-greedy", zorder=3, hatch='++', color='#EDB120')
    plt.ylabel('Average response time (s)')
    plt.xlabel('Link delays')
    plt.xticks(x, labels)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(linestyle='--', zorder=0)
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    plt.savefig("linkdelaybar_smallfont.eps", format='eps')
    plt.savefig("linkdelaybar.png", format='png')
    plt.show()
def prodelaybar():
    d1Cost = [float(r) for r in open("Costrd.txt", 'r').readlines()]
    d1dyg = [float(r) for r in open("dygCostrd.txt", 'r').readlines()]
    d1eg = [float(r) for r in open("egCostrd.txt", 'r').readlines()]
    d2Cost = [float(r) for r in open("Costc1.txt", 'r').readlines()]
    d2dyg = [float(r) for r in open("dygCostc1.txt", 'r').readlines()]
    d2eg = [float(r) for r in open("egCostc1.txt", 'r').readlines()]
    d3Cost = [float(r) for r in open("Costc2.txt", 'r').readlines()]
    d3dyg = [float(r) for r in open("dygCostc2.txt", 'r').readlines()]
    d3eg = [float(r) for r in open("egCostc2.txt", 'r').readlines()]

    Y1 = [d1Cost, d2Cost, d3Cost]
    Y2 = [d1dyg, d2dyg, d3dyg]
    Y3 = [d1eg, d2eg, d3eg]

    labels = ("10-50ms", "50-100ms", "100-150ms")
    x = np.arange(len(labels))
    y1 = [sum(y) / len(y) for y in Y1]
    e1 = [mean_confidence_interval(y) for y in Y1]
    y2 = [sum(y) / len(y) for y in Y2]
    e2 = [mean_confidence_interval(y) for y in Y2]
    y3 = [sum(y) / len(y) for y in Y3]
    e3 = [mean_confidence_interval(y) for y in Y3]
    width = 0.3
    plt.bar(x - width, y1, width, yerr=e1, capsize=7, label="BandEdge", zorder=3, hatch='//', color='#4DBEEE')
    plt.bar(x, y2, width, yerr=e2, capsize=7, label="adaptive greedy", zorder=3, hatch='..', color='#A2142F')
    plt.bar(x + width, y3, width, yerr=e3, capsize=7, label="$\epsilon$-greedy", zorder=3, hatch='++', color='#EDB120')
    plt.ylabel('Average response time (s)')
    plt.xlabel('Processing delays')
    plt.xticks(x, labels)
    plt.grid(linestyle='--', zorder=0)
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    plt.savefig("prodelaybar_small font.eps", format='eps')
    plt.savefig("prodelaybar.png", format='png')
    plt.show()
def switchdelaybar():
    d1Cost = [float(r) for r in open("Costcd1.txt", 'r').readlines()]
    d1dyg = [float(r) for r in open("dygCostcd1.txt", 'r').readlines()]
    d1eg = [float(r) for r in open("egCostcd1.txt", 'r').readlines()]
    d2Cost = [float(r) for r in open("Costcd2.txt", 'r').readlines()]
    d2dyg = [float(r) for r in open("dygCostcd2.txt", 'r').readlines()]
    d2eg = [float(r) for r in open("egCostcd2.txt", 'r').readlines()]
    d3Cost = [float(r) for r in open("Costcd3.txt", 'r').readlines()]
    d3dyg = [float(r) for r in open("dygCostcd3.txt", 'r').readlines()]
    d3eg = [float(r) for r in open("egCostcd3.txt", 'r').readlines()]

    Y1 = [d1Cost, d2Cost, d3Cost]
    Y2 = [d1dyg, d2dyg, d3dyg]
    Y3 = [d1eg, d2eg, d3eg]

    labels = ("10-50ms", "50-100ms", "100-150ms")
    x = np.arange(len(labels))
    y1 = [sum(y) / len(y) for y in Y1]
    e1 = [mean_confidence_interval(y) for y in Y1]
    y2 = [sum(y) / len(y) for y in Y2]
    e2 = [mean_confidence_interval(y) for y in Y2]
    y3 = [sum(y) / len(y) for y in Y3]
    e3 = [mean_confidence_interval(y) for y in Y3]
    width = 0.3
    plt.bar(x - width, y1, width, yerr=e1, capsize=7, label="BandEdge", zorder=3, hatch='//', color='#4DBEEE')
    plt.bar(x, y2, width, yerr=e2, capsize=7, label="$\epsilon$-greedy", zorder=3, hatch='..', color='#A2142F')
    plt.bar(x + width, y3, width, yerr=e3, capsize=7, label="adaptive greedy", zorder=3, hatch='++', color='#EDB120')
    plt.ylabel('Average response time (s)', fontsize=15)
    plt.xlabel('Processing delays', fontsize=15)
    plt.xticks(x, labels)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    plt.grid(linestyle='--')
    plt.savefig("switchdelaybar_smallfont.eps", format='eps')
    plt.savefig("switchdelaybar.png", format='png')
    plt.show()
def numServicebarMininet():
    s2cost = [float(r) for r in open("Costs2.txt", 'r').readlines()]
    s2ag = [float(r) for r in open("dygCosts2.txt", 'r').readlines()]
    s2eg = [float(r) for r in open("egCosts2.txt", 'r').readlines()]
    s3cost = [float(r) for r in open("Costs3.txt", 'r').readlines()]
    s3ag = [float(r) for r in open("dygCosts3.txt", 'r').readlines()]
    s3eg = [float(r) for r in open("egCosts3.txt", 'r').readlines()]
    s4cost = [float(r) for r in open("Costs4.txt", 'r').readlines()]
    s4ag = [float(r) for r in open("dygCosts4.txt", 'r').readlines()]
    s4eg = [float(r) for r in open("egCosts4.txt", 'r').readlines()]
    Y1 = [s2cost, s3cost, s4cost]
    Y2 = [s2eg, s3eg, s4eg]
    Y3 = [s2ag, s3ag, s4ag]

    y1 = [sum(y) / len(y) for y in Y1]
    e1 = [mean_confidence_interval(y) for y in Y1]
    y2 = [sum(y) / len(y) for y in Y2]
    e2 = [mean_confidence_interval(y) for y in Y2]
    y3 = [sum(y) / len(y) for y in Y3]
    e3 = [mean_confidence_interval(y) for y in Y3]

    labels = ["2", "3", "4"]
    x = np.arange(len(labels))
    width = 0.3
    plt.bar(x - width, y1, width, yerr=e1, capsize=7, label="BandEdge", zorder=3, hatch='//', color='#4DBEEE')
    plt.bar(x, y3, width, yerr=e2, capsize=7, label="adaptive greedy", zorder=3, hatch='..', color='#A2142F')
    plt.bar(x + width, y2, width, yerr=e3, capsize=7, label="$\epsilon$-greedy", zorder=3, hatch='++', color='#EDB120')
    plt.ylabel('Average response time (s)')
    plt.xlabel('Length of SFC')
    plt.xticks(x, labels)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(linestyle='--', zorder=0)
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    plt.savefig("numservicebarmininet_smallfont.eps", format='eps')
    plt.savefig("numservicebarmininet.png", format='png')
    plt.show()
def numServicebar():
    s2cost = ast.literal_eval(open("cmabn17s2", 'r').readline())
    s2eg = ast.literal_eval(open("egreedyn17s2", 'r').readline())
    s2dyg = ast.literal_eval(open("agreedyn17s2", 'r').readline())
    s3cost = ast.literal_eval(open("cmabn17s3", 'r').readline())
    s3eg = ast.literal_eval(open("egreedyn17s3", 'r').readline())
    s3dyg = ast.literal_eval(open("agreedyn17s3", 'r').readline())
    s4cost = ast.literal_eval(open("cmabn17s4", 'r').readline())
    s4eg = ast.literal_eval(open("egreedyn17s4", 'r').readline())
    s4dyg = ast.literal_eval(open("agreedyn17s4", 'r').readline())
    s5cost = ast.literal_eval(open("cmabn17s5", 'r').readline())
    s5eg = ast.literal_eval(open("egreedyn17s5", 'r').readline())
    s5dyg = ast.literal_eval(open("agreedyn17s5", 'r').readline())
    Y1 = [s2cost, s3cost, s4cost, s5cost]
    Y2 = [s2eg, s3eg, s4eg, s5eg]
    Y3 = [s2dyg, s3dyg, s4dyg, s5dyg]

    y1 = [sum(y) / len(y) for y in Y1]
    y2 = [sum(y) / len(y) for y in Y2]
    y3 = [sum(y) / len(y) for y in Y3]
    y = y1 + y2 + y3
    N = max(y) - min(y)
    y1 = [y/N for y in y1]
    y2 = [y/N for y in y2]
    y3 = [y/N for y in y3]
    e1 = [mean_confidence_interval(y)/N for y in Y1]
    e2 = [mean_confidence_interval(y)/N for y in Y2]
    e3 = [mean_confidence_interval(y)/N for y in Y3]

    labels = ["2", "3", "4", "5"]
    x = np.arange(len(labels))
    width = 0.3
    plt.bar(x - width, y1, width, yerr=e1, capsize=7, label="BandEdge", zorder=3, hatch='//', color='#4DBEEE')
    plt.bar(x, y2, width, yerr=e2, capsize=7, label="$\epsilon$-greedy", zorder=3, hatch='..', color='#A2142F')
    plt.bar(x + width, y3, width, yerr=e3, capsize=7, label="adaptive greedy", zorder=3, hatch='++', color='#EDB120')
    plt.ylabel('Time average cost', fontsize=15)
    plt.xlabel('Number of services on SFC', fontsize=15)
    plt.xticks(x, labels)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(linestyle='--', zorder=0)
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    plt.savefig("numservicebar.eps", format='eps')
    plt.savefig("numservicebar.png", format='png')
    plt.show()
def numNodebar():
    n5cost = ast.literal_eval(open("cmabn5s3", 'r').readline())
    n5eg  = ast.literal_eval(open("egreedyn5s3", 'r').readline())
    n5dyg = ast.literal_eval(open("agreedyn5s3", 'r').readline())
    n10cost = ast.literal_eval(open("cmabn10s3", 'r').readline())
    n10eg = ast.literal_eval(open("egreedyn10s3", 'r').readline())
    n10dyg = ast.literal_eval(open("agreedyn10s3", 'r').readline())
    n17cost = ast.literal_eval(open("cmabn17s3", 'r').readline())
    n17eg = ast.literal_eval(open("egreedyn17s3", 'r').readline())
    n17dyg = ast.literal_eval(open("agreedyn17s3", 'r').readline())
    n26cost = ast.literal_eval(open("cmabn26s3", 'r').readline())
    n26eg = ast.literal_eval(open("egreedyn26s3", 'r').readline())
    n26dyg = ast.literal_eval(open("agreedyn26s3", 'r').readline())
    Y1 = [n5cost, n10cost, n17cost, n26cost]
    Y2 = [n5eg, n10eg, n17eg, n26eg]
    Y3 = [n5dyg, n10dyg, n17dyg, n26dyg]

    y1 = [sum(y) / len(y) for y in Y1]
    e1 = [mean_confidence_interval(y) for y in Y1]
    y2 = [sum(y) / len(y) for y in Y2]
    e2 = [mean_confidence_interval(y) for y in Y2]
    y3 = [sum(y) / len(y) for y in Y3]
    e3 = [mean_confidence_interval(y) for y in Y3]
    y = y1 + y2 + y3
    N = max(y) - min(y) + 70
    y1 = [y / N for y in y1]
    y2 = [y / N for y in y2]
    y3 = [y / N for y in y3]
    e1 = [mean_confidence_interval(y) / N for y in Y1]
    e2 = [mean_confidence_interval(y) / N for y in Y2]
    e3 = [mean_confidence_interval(y) / N for y in Y3]

    labels = ["2x2", "3x3", "4x4", "5x5"]
    x = np.arange(len(labels))
    width = 0.3
    plt.bar(x - width, y1, width, yerr=e1, capsize=7, label="BandEdge", zorder=3, hatch='//', color='#4DBEEE')
    plt.bar(x, y2, width, yerr=e2, capsize=7, label="$\epsilon$-greedy", zorder=3, hatch='..', color='#A2142F')
    plt.bar(x + width, y3, width, yerr=e3, capsize=7, label="adaptive greedy", zorder=3, hatch='++', color='#EDB120')
    plt.ylabel('Time average cost', fontsize=15)
    plt.xlabel('Size of the MEC network', fontsize=15)
    plt.xticks(x, labels)
    plt.grid(linestyle='--', zorder=0)
    plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)
    plt.grid(linestyle='--')
    plt.savefig("numnodebar.eps", format='eps')
    plt.savefig("numnodebar.png", format='png')
    plt.show()
def readdata():
    f = open("delayvscost.txt", "r")
    y = []
    err = []
    for d in f:
        y.append(float(d.split()[0]))
        err.append(float(d.split()[1]))
    return y, err
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h
def calculateoffline(n):
    e = sys.argv[1]
    f = open("estimation%s.txt"%e, 'r')
    fstr = f.readline()
    T = ast.literal_eval(fstr)
    # T = [[[0.01], [0.45], [0.1], [0.25], [1.0], [0.6]],
    #         [[0.45], [0.02], [0.35], [0.2], [0.95], [0.55]],
    #         [[0.1], [0.35], [0.03], [0.15], [0.9], [0.5]],
    #         [[0.25], [0.2], [0.15], [0.04], [0.75], [0.35]],
    #         [[1.0], [0.95], [0.9], [0.75], [0.05], [0.4]],
    #         [[0.6], [0.55], [0.5], [0.35], [0.4], [0.06]]]
    c = [0.0 for i in range(n)]  #
    f = open("Demand%s.txt"%e, 'r')
    fstr = f.readline()
    d = ast.literal_eval(fstr)
    f = open("Posd%s.txt"%e, 'r')
    fstr = f.readline()
    p = ast.literal_eval(fstr)
    for i in range(n):
        c[i] = sum(T[i][i]) / len(T[i][i])
    f = [[0.0 for j in range(n)] for k in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                f[i][j] = 0.0
            else:
                f[i][j] = sum(T[i][j]) / len(T[i][j])
    lp = random.sample([i for i in range(n)], service_num)
    o = offline(None, len(d), [0]+d, [0]+p, f, c, lp)/len(d)
    f = open("offline%s.txt"%e, 'w')
    f.write(str(o))
    f.close()
    return o

if __name__ == '__main__':
    #replot2y()
    # numServicebarMininet()
    # linkdelaybar()
    # prodelaybar()
    # mobilitybar()
    #switchdelaybar()
    #replotRuntime()
    # #numNodebar()
    # #numServicebar()
    # # plotComputingWeight(None)
    plotSwitchCost(None)
    #plotTransWeight(None)
    # #replotMiniWifiLearning()
    # #replotRuntime()
    # #replot()
    # #numNodebar()
    # #numServicebarMininet()
    # #numServicebar()
    # #iters = 1000
    # # # for er in [0.01, 0.05, 0.09]:
    # # #     explore_ratio = er
    # # #     service_num = 3
    # # #     G = createNetwork(5, 3, 3)
    # # #     mainLearning(G, iters, 0.2)
    # #
    # # for n in [3, 4, 5, 6]:
    # #     RT = []
    # #     for sn in [3, 4, 5, 6, 7, 8, 9, 10]:
    # #         service_num = sn
    # #         G = createNetwork(n, 3, 3)
    # #         rt = mainLearning(G, iters, 0.2)
    # #         RT.append(rt)
    # #     open("runtimeN%i"%n, "w").write(str(RT))
    # # explore_ratio = 0.0
    # # service_num = 3
    # #G = createNetwork(5, 3, 3)
    #
    #
    # #print(G.edges(data='weight'))
    # # # #plotExploreratio(G, 10000)
    # #mainLearning(G, iters, 0.2)
    # #plotRegret(iters)
    # #plotRatio(iters)
    # #replotregret()
    # # plotLearningSlot(5, 3, 3, None)
    # # plotTransWeight(500, None)
    # # plotComputingWeight(500)
    # # plotSwitchCost(500)
    #
    # # #plotServiceNumRuntime(10, 10, 3)
    # # labels = [0.05, 0.1, 0.15, 0.2, 0.25]
    # # for sw in labels:
    # #     global switch_weight
    # #     switch_weight = sw
    # #     G = createNetwork(5, 0, 1)
    # #     cmab, cloud, edge, offline = main2(G, 100)
    #
    # # for th in [50, 60, 70, 80, 90, 100]:
    # #     plotSwitchCost(100)
    # # for t in range(5):
    # #     for c in range(5):
    # #         G = createNetwork(5, t, c)
    # #         main2(G, 100)
    # # mobilitybar()
    # #delaybar()
    # #mobilitybar()
    # #linkdelaybar()
    # #prodelaybar()
    # #numServicebar()
    # # explore_search_range()
    # #print(calculateoffline(6))









