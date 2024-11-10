
---
# Chapter 2. Routing Information Protocol (RIP)

RIP is the first in a family of dynamic routing protocols that we will look at closely. Dynamic routing protocols _automatically_ compute routing tables_,_ freeing the network administrator from the task of specifying routes to every network using static routes. Indeed, given the complexity of and number of routes in most networks, static routing usually is not even an option.

In addition to computing the “shortest” paths to all destination networks, dynamic routing protocols discover alternative (second-best) paths when a primary path fails and balance traffic over multiple paths ( load balancing).

Most dynamic routing protocols are based on one of two distributed algorithms: Distance Vector or Link State. RIP, upon which Cisco’s IGRP was based, is a classic example of a DV protocol. Link State protocols include OSPF, which we will look at in a later chapter. The following section gets us started with configuring RIP.

# Getting RIP Running

Throughout this book, we’ll be using a fictional network called TraderMary to illustrate the concepts with which we’re working. TraderMary is a distributed network with nodes in New York, Chicago, and Ames, Iowa, as shown in Figure 2-1

![[Pasted image 20241110162658.png]]

As a distributed process, RIP needs to be configured on every router in the network:
```
hostname NewYork
...
interface Ethernet0
ip address 172.16.1.1 255.255.255.0
!
interface Ethernet1
ip address 192.168.1.1 255.255.255.0
!
interface Serial0
ip address 172.16.250.1 255.255.255.0
!
interface Serial1
ip address 172.16.251.1 255.255.255.0
...
router rip
network 172.16.0.0


hostname Chicago
...
interface Ethernet0
ip address 172.16.50.1 255.255.255.0
!
interface Serial0
ip address 172.16.250.2 255.255.255.0
!
interface Serial1
ip address 172.16.252.1 255.255.255.0
...

router rip
network 172.16.0.0


hostname Ames
...
interface Ethernet0
ip address 172.16.100.1 255.255.255.0
!
interface Serial0
ip address 172.16.252.2 255.255.255.0
!
interface Serial1
ip address 172.16.251.2 255.255.255.0
...

router rip
network 172.16.0.0
```

Notice that all that is required of a network administrator to start RIP on a router is to issue the following command:

```
router rip
```

in global configuration mode and to list the networks that will be participating in the RIP process:

```
network 172.16.0.0
```

What does it mean to list the network numbers participating in RIP?

1. Router _NewYork_ will include directly connected `172.16.0.0` subnets in its updates to neighboring routers. For example, `172.16.1.0` will now be included in updates to the routers _Chicago_ and _Ames_.
    
2. _NewYork_ will receive and process RIP updates on its `172.16.0.0` interfaces from other routers running RIP. For example, _NewYork_ will receive RIP updates from _Chicago_ and _Ames_.
    
3. By exclusion, network `192.168.1.0`, connected to _NewYork_, will not be advertised to _Chicago_ or _Ames_, and _NewYork_ will not process any RIP updates received on _Ethernet0_ (if there is another router on that segment)

Next, let’s verify that all the routers are seeing all the `172.16.0.0` subnets:

```
   NewYork>sh ip route
   Codes: C - connected, S - static, I - IGRP, R - RIP, M - mobile, B - BGP
          D - EIGRP, EX - EIGRP external, O - OSPF, IA - OSPF inter area
          N1 - OSPF NSSA external type 1, N2 - OSPF NSSA external type 2
          E1 - OSPF external type 1, E2 - OSPF external type 2, E - EGP
          i - IS-IS, L1 - IS-IS level-1, L2 - IS-IS level-2, * - candidate default

   Gateway of last resort is not set

   C       192.168.1.0 is directly connected, Ethernet1
        172.16.0.0/16 is subnetted, 6 subnets
   C       172.16.1.0 is directly connected, Ethernet0
   C       172.16.250.0 is directly connected, Serial0
   C       172.16.251.0 is directly connected, Serial1
   R       172.16.50.0 [120/1] via 172.16.250.2, 0:00:11, Serial0
   R       172.16.100.0 [120/1] via 172.16.251.2, 0:00:19, Serial1
   R       172.16.252.0 [120/1] via 172.16.250.2, 0:00:11, Serial0
                        [120/1] via 172.16.251.2, 0:00:19, Serial1


   Chicago>sh ip route
   Codes: C - connected, S - static, I - IGRP, R - RIP, M - mobile, B - BGP
          D - EIGRP, EX - EIGRP external, O - OSPF, IA - OSPF inter area
          N1 - OSPF NSSA external type 1, N2 - OSPF NSSA external type 2
          E1 - OSPF external type 1, E2 - OSPF external type 2, E - EGP
          i - IS-IS, L1 - IS-IS level-1, L2 - IS-IS level-2, * - candidate default

   Gateway of last resort is not set

        172.16.0.0/16 is subnetted, 6 subnets
   C       172.16.50.0 is directly connected, Ethernet0
   C       172.16.250.0 is directly connected, Serial0
   C       172.16.252.0 is directly connected, Serial1
   R       172.16.1.0 [120/1] via 172.16.250.1, 0:00:01, Serial0
   R       172.16.100.0 [120/1] via 172.16.252.2, 0:00:10, Serial1
   R       172.16.251.0 [120/1] via 172.16.250.1, 0:00:01, Serial0
                       [120/1] via 172.16.252.2, 0:00:10, Serial1


   Ames>sh ip route
   Codes: C - connected, S - static, I - IGRP, R - RIP, M - mobile, B - BGP
          D - EIGRP, EX - EIGRP external, O - OSPF, IA - OSPF inter area
          N1 - OSPF NSSA external type 1, N2 - OSPF NSSA external type 2
          E1 - OSPF external type 1, E2 - OSPF external type 2, E - EGP
          i - IS-IS, L1 - IS-IS level-1, L2 - IS-IS level-2, * - candidate default

   Gateway of last resort is not set

1       **`172.16.0.0/16 is subnetted, 6 subnets`**
   C       172.16.100.0 is directly connected, Ethernet0
   C       172.16.252.0 is directly connected, Serial0
   C       172.16.251.0 is directly connected, Serial1
   R       172.16.50.0 [120/1] via 172.16.252.1, 0:00:21, Serial0
   R       172.16.1.0 [120/1] via 172.16.251.1, 0:00:09, Serial1
   R       172.16.250.0 [120/1] via 172.16.252.1, 0:00:21, Serial0
                        [120/1] via 172.16.251.1, 0:00:09, Serial1
```

The left margin in the output of the routing tables shows how the route was derived. "C” indicates a directly connected network; “R” indicates RIP. Further note that there is some indentation in the output. The subnets of `172.16.0.0` are indented under line 1, which gives us the number of subnets (6) in `172.16.0.0` and the subnet mask that is associated with this network (`/16`). The routing table provides this information for every major network number it knows, indenting the subnets below the major network number.

Configuring RIP is fairly straightforward. We’ll examine how RIP works in more detail in the next section.