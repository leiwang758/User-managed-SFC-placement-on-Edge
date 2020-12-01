#!/usr/bin/python

'Example for Handover'

import sys

import random
from mininet.node import RemoteController, Controller
from mininet.log import setLogLevel, info
from mn_wifi.cli import CLI_wifi
from mn_wifi.net import Mininet_wifi
# from mininet.link import TCLink
from mn_wifi.associationControl import associationControl


n = 9



def topology(args):
    Host = []
    Ap = []

    "Create a network."
    net = Mininet_wifi(controller=RemoteController)

    if '-lv' in sys.argv:
        usr = net.addStation('usr', mac='00:00:00:00:00:01', ip='10.0.0.1/8', range=5, min_v=1, max_v=5)
    elif '-mv' in sys.argv:
        usr = net.addStation('usr', mac='00:00:00:00:00:01', ip='10.0.0.1/8', range=5, min_v=5, max_v=10)
    elif '-hv' in sys.argv:
        usr = net.addStation('usr', mac='00:00:00:00:00:01', ip='10.0.0.1/8', range=5, min_v=10, max_v=15)
    else:
        usr = net.addStation('usr', mac='00:00:00:00:00:01', ip='10.0.0.1/8', range=5, min_x=0, max_x=90, min_y=0, max_y=90, min_v=1, max_v=5, min_wt=0, max_wt=5, constantDistance=1, constantVelocity=1)
    for i in range(n):
        Host.append(net.addHost('h%i'%(i+1), mac='00:00:00:00:00:0%i'%(i+2), ip='10.0.0.%i/8'%(i+2)))

    Host.append(net.addHost('cloud', mac="0:00:00:00:00:0%i"%(n+2), ip='10.0.0.%i'%(n+2)))

    info("*** Creating nodes\n")
    # if '-s' in args:
    #     sta1 = net.addStation('sta1', mac='00:00:00:00:00:02', ip='10.0.0.2/8',
    #                           position='20,30,0')
    #     sta2 = net.addStation('sta2', mac='00:00:00:00:00:03', ip='10.0.0.3/8',
    #                           position='60,30,0')
    # else:
    #     sta1 = net.addStation('sta1', mac='00:00:00:00:00:02', ip='10.0.0.2/8',
    #                           range=20)
    #     sta2 = net.addStation('sta2', mac='00:00:00:00:00:03', ip='10.0.0.3/8',
    #                           range=20)
    ap1 = net.addAccessPoint('ap1', ssid='ssid-ap1', mode='g', channel='1',
                             position='15,15,0', range=25)
    ap2 = net.addAccessPoint('ap2', ssid='ssid-ap2', mode='g', channel='2',
                             position='45,15,0', range=25)
    ap3 = net.addAccessPoint('ap3', ssid='ssid-ap3', mode='g', channel='3',
                             position='15,45,0', range=25)
    ap4 = net.addAccessPoint('ap4', ssid='ssid-ap4', mode='g', channel='4',
                             position='45,45,0', range=25)
    ap5 = net.addAccessPoint('ap5', ssid='ssid-ap5', mode='g', channel='5',
                             position='15,75,0', range=25)
    ap6 = net.addAccessPoint('ap6', ssid='ssid-ap6', mode='g', channel='6',
                             position='45,75,0', range=25)
    ap7 = net.addAccessPoint('ap7', ssid='ssid-ap7', mode='g', channel='7',
                             position='75,15,0', range=25)
    ap8 = net.addAccessPoint('ap8', ssid='ssid-ap8', mode='g', channel='8',
                             position='75,45,0', range=25)
    ap9 = net.addAccessPoint('ap9', ssid='ssid-ap9', mode='g', channel='9',
                             position='75,75,0', range=25)
    c1 = net.addController('c1', controller=RemoteController, ip='127.0.0.1', port=6633)
    #c1 = net.addController('c1')
    Ap = [ap1, ap2, ap3, ap4, ap5, ap6, ap7, ap8, ap9]

    net.setPropagationModel(model="logDistance", exp=5)



    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()


    if "-d1" in sys.argv:
        a = 50
        b = 100
    elif "-d2" in sys.argv:
        a = 150
        b = 200
    elif "-d3" in sys.argv:
        a = 250
        b = 300
    elif "-d4" in sys.argv:
        a = 350
        b = 400
    elif "-d5" in sys.argv:
        a = 450
        b = 500
    elif "-D" in sys.argv:
        a = 1000
        b = 2000
    else:
        a = 100
        b = 500

    info("*** Creating links\n")
    net.addLink(ap1, ap3, delay='%ims'%(random.randint(a, b)))
    net.addLink(ap3, ap4, delay='%ims'%(random.randint(a, b)))
    net.addLink(ap2, ap4, delay='%ims'%(random.randint(a, b)))
    net.addLink(ap2, ap7, delay='%ims'%(random.randint(a, b)))
    net.addLink(ap4, ap8, delay='%ims'%(random.randint(a, b)))
    net.addLink(ap4, ap6, delay='%ims'%(random.randint(a, b)))
    net.addLink(ap5, ap6, delay='%ims'%(random.randint(a, b)))
    net.addLink(ap8, ap9, delay='%ims'%(random.randint(a, b)))

    # net.addLink(ap1, ap3, delay='100ms')
    # net.addLink(ap3, ap4, delay='150ms')
    # net.addLink(ap2, ap4, delay='200ms')
    # net.addLink(ap2, ap7, delay='250ms')
    # net.addLink(ap4, ap8, delay='300ms')
    # net.addLink(ap4, ap6, delay='350ms')
    # net.addLink(ap5, ap6, delay='400ms')
    # net.addLink(ap8, ap9, delay='450ms')



    for i in range(n):
        net.addLink(Host[i], Ap[i], bw=100, delay='1ms', loss=2)
    net.addLink(Host[-1], Ap[3], bw=100, delay='%ims'%(random.randint(100, 300)), loss=2)


    net.plotGraph(max_x=100, max_y=100)
    net.setAssociationCtrl('ssf')
    if "-ssf" in sys.argv:
        net.setAssociationCtrl('ssf')
    elif "-llf" in sys.argv:
        net.setAssociationCtrl('llf')


    if '-rw' in sys.argv:
        net.setMobilityModel(time=0, model='RandomWalk', max_x=60, max_y=90,
                             seed=20)
    elif '-rd' in sys.argv:
        net.setMobilityModel(time=0, model='RandomDirection', max_x=60, max_y=90,
                             seed=20)
    elif '-rwp' in sys.argv:
        net.setMobilityModel(time=0, model='RandomWayPoint', max_x=60, max_y=90,
                             seed=20)
    elif '-GM' in sys.argv:
        net.setMobilityModel(time=0, model='GaussMarkov', max_x=60, max_y=90,
                             seed=20)
    elif '-TVC' in sys.argv:
        net.setMobilityModel(time=0, model='TimeVariantCommunity', max_x=60, max_y=90,
                             seed=20)
    elif '-RP' in sys.argv:
        net.setMobilityModel(time=0, model='ReferencePoint', max_x=60, max_y=90,
                             seed=20)
    elif '-TLW' in sys.argv:
        net.setMobilityModel(time=0, model='TruncatedLevyWalk', max_x=60, max_y=90,
                             seed=20)
    else:
        net.setMobilityModel(time=0, model='RandomDirection', max_x=60, max_y=90,
                             seed=20)

    info("*** Starting network\n")
    net.build()
    c1.start()
    for ap in Ap:
        ap.start([c1])
        #ap2.start([c1])

    info("*** Running CLI\n")
    CLI_wifi(net)

    info("*** Stopping network\n")
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    topology(sys.argv)
