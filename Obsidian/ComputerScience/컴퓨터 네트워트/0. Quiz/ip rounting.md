
---
# Chapter 1. Routing Information Protocol (RIP)

RIP is the first in a family of dynamic routing protocols that we will look at closely. Dynamic routing protocols _automatically_ compute routing tables_,_ freeing the network administrator from the task of specifying routes to every network using static routes. Indeed, given the complexity of and number of routes in most networks, static routing usually is not even an option.

In addition to computing the “shortest” paths to all destination networks, dynamic routing protocols discover alternative (second-best) paths when a primary path fails and balance traffic over multiple paths ( load balancing).

Most dynamic routing protocols are based on one of two distributed algorithms: Distance Vector or Link State. RIP, upon which Cisco’s IGRP was based, is a classic example of a DV protocol. Link State protocols include OSPF, which we will look at in a later chapter. The following section gets us started with configuring RIP.

## Getting RIP Running

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

1       `172.16.0.0/16 is subnetted, 6 subnets`
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

## How RIP Finds Shortest Paths

All DV protocols essentially operate the same way: routers exchange routing updates with neighboring (directly connected) routers; the routing updates contain a list of network numbers along with the distance (metric, in routing terminology) to the networks. Each router chooses the shortest path to a destination network by comparing the distance (or metric) information it receives from its various neighbors. Let’s look at this in more detail in the context of RIP.

Let’s imagine that the network is cold-started -- i.e., all three routers are powered up at the same time. The first thing that happens after IOS has finished loading is that the router checks for its connected interfaces and determines which ones are up. Next, these directly connected networks are installed in each router’s routing table. So, right after IOS has been loaded and before any routing updates have been exchanged, the routing table would look like this:

```
NewYork>sh ip route
Codes: C - connected, S - static, I - IGRP, R - RIP, M - mobile, B - BGP
       D - EIGRP, EX - EIGRP external, O - OSPF, IA - OSPF inter area
       N1 - OSPF NSSA external type 1, N2 - OSPF NSSA external type 2
       E1 - OSPF external type 1, E2 - OSPF external type 2, E - EGP
       i - IS-IS, L1 - IS-IS level-1, L2 - IS-IS level-2, * - candidate default

Gateway of last resort is not set

C       171.16.1.0 is directly connected, Ethernet0
C       171.16.250.0 is directly connected, Serial0
C       171.16.251.0 is directly connected, Serial1


Chicago>sh ip route
Codes: C - connected, S - static, I - IGRP, R - RIP, M - mobile, B - BGP
       D - EIGRP, EX - EIGRP external, O - OSPF, IA - OSPF inter area
       N1 - OSPF NSSA external type 1, N2 - OSPF NSSA external type 2
       E1 - OSPF external type 1, E2 - OSPF external type 2, E - EGP
       i - IS-IS, L1 - IS-IS level-1, L2 - IS-IS level-2, * - candidate default

Gateway of last resort is not set

C       171.16.50.0 is directly connected, Ethernet0
C       171.16.250.0 is directly connected, Serial0
C       171.16.252.0 is directly connected, Serial1


Ames>sh ip route
Codes: C - connected, S - static, I - IGRP, R - RIP, M - mobile, B - BGP
       D - EIGRP, EX - EIGRP external, O - OSPF, IA - OSPF inter area
       N1 - OSPF NSSA external type 1, N2 - OSPF NSSA external type 2
       E1 - OSPF external type 1, E2 - OSPF external type 2, E - EGP
       i - IS-IS, L1 - IS-IS level-1, L2 - IS-IS level-2, * - candidate default

Gateway of last resort is not set

C       171.16.100.0 is directly connected, Ethernet0
C       171.16.250.0 is directly connected, Serial0
C       171.16.252.0 is directly connected, Serial1
```

The routers are now ready to update their neighbors with these routes.

### RIP Update

RIP updates are encapsulated in UDP. The well-known port number for RIP updates is 520. The format of a RIP packet is shown in Figure 2-2

![[Pasted image 20241110163305.png]]

Figure 2-2. Format of RIP update packet

Note that RIP allows a station to request routes, so a machine that has just booted up can request the routing table from its neighbors instead of waiting until the next cycle of updates.

The destination IP address in RIP updates is `255.255.255.255`. The source IP address is the IP address of the interface from which the update is issued.

If you look closely at the update you will see that a key piece of information is missing: the subnet mask. Let’s say that an update was received with the network number `172.31.0.0`. Should this be interpreted as `172.31.0.0/16` or `172.31.0.0/24` or `172.31.0.0/26` or ...? This question is addressed later, in [Section 2.4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s04.html "Subnet Masks").

### RIP Metric

The RIP metric is simply a measure of the number of hops to a destination network. `172.16.100.0`, which is directly connected to _Ames_, is zero hops from _Ames_ but one hop from _NewYork_ and _Chicago_. You can see RIP metrics in the routing table:
```
NewYork>sh ip route
...
Gateway of last resort is not set

C       192.168.1.0 is directly connected, Ethernet1
     172.16.0.0/16 is subnetted, 6 subnets
C       172.16.1.0 is directly connected, Ethernet0
C       172.16.250.0 is directly connected, Serial0
C       172.16.251.0 is directly connected, Serial1
R       172.16.50.0 **`[120/1]`** via 172.16.250.2, 0:00:11, Serial0
R       172.16.100.0 **`[120/1]`** via 172.16.251.2, 0:00:19, Serial1
R       172.16.252.0 **`[120/1]`** via 172.16.250.2, 0:00:11, Serial0
                     **`[120/1]`** via 172.16.251.2, 0:00:19, Serial1
```

This routing table shows the [distance/metric] tuple in bold. Every hop between two routers adds 1 to the RIP metric. Thus, _NewYork_ sees the _Ames_ segment (`172.16.100.0`) as one hop via the direct 56-kbps link and two hops via the T-1 to _Chicago_. _NewYork_ will prefer the direct one-hop path to _Ames_.

The simplicity of the RIP metric is an asset in small, homogenous networks but a liability in networks with heterogeneous media. Consider the following comparison: the transmission delay for a 1,000-octet packet is 143 ms over a 56-kbps link and 5 ms over a T-1 link. Neglecting buffering and processing delays, two T-1 hops will cost 10 ms in comparison to 143 ms via the 56-kbps link. Thus, the two-hop T-1 path between _NewYork_ and _Ames_ is quicker; indeed, the designers of TraderMary’s network may have put in the 56-kbps link only for backup purposes. However, RIP does not account for line speed, delay, or reliability. For this, we will look to the next DV protocol -- IGRP.

Let’s look at one more example of RIP metrics for TraderMary’s network. Let’s say that the T-1 link between _NewYork_ and _Chicago_ fails. As soon as _NewYork_ (or _Chicago_) detects a failure in the link, all routes associated with that link are purged from the routing table, and, upon receipt of the next update, _NewYork_ (_Chicago_) will learn the routes to _Chicago_ (_NewYork_) via _Ames_. _NewYork_’s routing table would look like this:

```
NewYork>sh ip route
...
Gateway of last resort is not set

C       192.168.1.0 is directly connected, Ethernet1
     172.16.0.0/16 is subnetted, 6 subnets
C       172.16.1.0 is directly connected, Ethernet0
C       172.16.251.0 is directly connected, Serial1
R       172.16.50.0 **`[120/2]`** via 172.16.251.2, 0:00:23, Serial1
R       172.16.100.0 **`[120/1]`** via 172.16.251.2, 0:00:23, Serial1
R       172.16.252.0 **`[120/1]`** via 172.16.251.2, 0:00:23, Serial1
```

As we discussed in the previous chapter, the distance value associated with RIP is 120. Note that directly connected routes do not show a distance or metric value. Directly connected routes have a distance value of and thus show the most preferred route to a destination, no matter how low the metric value of a route to the same network may be through another routing source (such as RIP).

The RIP metrics we saw in the previous examples were 1 or 2. It turns out that a RIP metric of 16 signals infinity (or unreachability). Why is it necessary to choose a maximum value for the RIP metric? Without a maximum hop count, a route can propagate indefinitely during certain failure scenarios, resulting in indefinitely long convergence times. This is discussed further in [Section 2.3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s03.html "Convergence") under [Section 2.3.1.2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s03.html#iprouting-CHP-2-SECT-3.1.2 "Counting to infinity").

### Processing RIP Updates

The following rules summarize the steps a router takes when it receives a RIP update:

1. If the destination network number is unknown to the router, install the route using the source IP address of the update (provided the hop count is less than 16).
    
2. If the destination network number is known to the router but the update contains a smaller metric, modify the routing table entry with the new next hop and metric.
    
3. If the destination network number is known to the router but the update contains a larger metric, ignore the update.
    
4. If the destination network number is known to the router and the update contains a higher metric that is from the same next hop as in the table, update the metric.
    
5. If the destination network number is known to the router and the update contains the same metric from a different next hop, RFC 1058 calls for this update to be ignored, in general. However, Cisco differs from the standard here and installs up to four parallel paths to the same destination. These parallel paths are then used for load balancing.
    

Thus, when the first update from _Ames_ reaches _NewYork_ with the network `172.16.100.0`, _NewYork_ installs the route with a hop count of 1 using rule 1. _NewYork_ will also receive `172.16.100.0` in a subsequent update from _Chicago_ (after _Chicago_ itself has learned the route from _Ames_), but _NewYork_ will discard this route because of rule 3.

### Steady State

It is important for you as the network administrator to be familiar with the state of the network during normal conditions. Deviations from this state will be your clue to troubleshooting the network during times of network outage.

The following output will show you the values of the RIP timers. Note that RIP updates are sent every 30 seconds and the next update is due in 24 seconds, which means that an update was issued about 6 seconds ago. We will discuss the invalid, hold-down, and flush timers later, in [Section 2.3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s03.html "Convergence").

```
NewYork>sh ip protocol

Routing Protocol is "rip"
Sending updates every 30 seconds, next due in 24 seconds
Invalid after 90 seconds, hold down 90, flushed after 180
```

One key area to look at in the routing table is the timer values. The format Cisco uses for timers is _hh:mm:ss_ (hours:minutes:seconds). You would expect the time against each route to be between and 30 seconds. If a route was received more than 30 seconds ago, that indicates a problem in the network. You should begin by checking to see if the next hop for the route is reachable. As an example, in line 1, `172.16.50.0` was learned 11 seconds ago from `172.16.250.2` (on _Serial0_).

```
NewYork>sh ip route
   ...
   Gateway of last resort is not set
 
   C       192.168.1.0 is directly connected, Ethernet1
        172.16.0.0/16 is subnetted, 6 subnets
   C       172.16.1.9 is directly connected, Ethernet0
   C       172.16.250.0 is directly connected, Serial0
   C       172.16.251.0 is directly connected, Serial1
2  R       172.16.50.0 [120/1] via 172.16.250.2, 0:00:11, Serial0
   R       172.16.100.0 [120/1] via 172.16.251.2, 0:00:19, Serial1
   R       172.16.252.0 [120/1] via 172.16.250.2, 0:00:11, Serial0
                        [120/1] via 172.16.251.2, 0:00:19, Serial1

```

### Parallel Paths

There are two equal-cost paths to network `172.16.252.0` from _NewYork_ -- one advertised by _Ames_ and the other by _Chicago_. _NewYork_ will install both routes in its routing table:

```
NewYork>sh ip route
...
Gateway of last resort is not set

C       192.168.1.0 is directly connected, Ethernet1
     172.16.0.0/16 is subnetted, 6 subnets
C       172.16.1.9 is directly connected, Ethernet0
C       172.16.250.0 is directly connected, Serial0
C       172.16.251.0 is directly connected, Serial1
R       172.16.50.0 [120/1] via 172.16.250.2, 0:00:11, Serial0
R       172.16.100.0 [120/1] via 172.16.251.2, 0:00:19, Serial1
R       172.16.252.0 [120/1] via 172.16.250.2, 0:00:11, Serial0
                     [120/1] via 172.16.251.2, 0:00:19, Serial1
```

Both paths are utilized to forward packets. How is traffic split over the two links? The answer depends on the switching mode configured on the Cisco router. Two common switching modes are process switching and fast switching.

### Process Switching

Process switching results in packet-by-packet load balancing -- one packet travels out on _serial0_ and the next packet travels out on _serial1_. Packet-by-packet load balancing is possible while process switching because in this switching mode the router examines its routing table for every packet it receives.

Process switching is configured as follows:

```
NewYork#-config#interface serial0
NewYork#-config-if#no ip route-cache
```

Packet switching is very CPU-intensive, as every packet causes a routing table lookup.

### Fast Switching

In this mode, only the first packet for a given destination is looked up in the routing table, and, as this packet is forwarded, its next hop (say, _serial0_) is placed in a cache. Subsequent packets for the same destination are looked up in the cache, not in the routing table. This implies that all packets for this destination will follow the same path (_serial0_).

Now, if another packet arrives that matches the routing entry `204.148.185.192`, it will be cached with a next hop of _serial1_. Henceforth, all packets to this second destination will follow _serial1_.

Fast switching thus load-balances destination-by-destination (or session-by-session). Fast switching is configured as follows:

```
NewYork#-config#interface serial0
NewYork#-config-if#ip route-cache
```

In fast switching, the first packet for a new destination causes a routing table lookup and the generation of a new entry in the route cache. Subsequent packets consult the route cache but not the routing table.

## Convergence

Changes -- planned and unplanned -- are normal in any network:

- A serial link breaks
    
- A new serial link is added to a network
    
- A router or hub loses power or malfunctions
    
- A new LAN segment is added to a network
    

All routers in the routing domain will not reflect these changes right away. This is because RIP routers rely on their direct neighbors for routing updates, which in turn rely on another set of neighbors. The routing process that is set into motion from the time of a network change (such as the failure of a link) until all routers correctly reflect the change is referred to as convergence. During convergence, routing connectivity between some parts of the network may be lost and, hence, an important question that is frequently asked is “How long will the network take to converge after such-and-such failure in the network?” The answer depends on a number of factors, including the network topology and the timers that have been defined for the routing protocol.

The following list defines the four timers that are key to the operation of any DV protocol, including RIP:

Update timer (default value: 30 seconds)

After sending a routing update, RIP sets the update timer to 0. When the timer expires, RIP issues another routing update. Thus, RIP updates are sent every 30 seconds.

Invalid timer (default value: 180 seconds)

Every time a router receives an update for a route, it sets the invalid timer to 0. The expiration of the invalid timer indicates that six consecutive updates were missed -- at this time, the source of the routing information is considered suspect. Even though the route is declared invalid, packets are still forwarded to the next hop specified in the routing table. Note that prior to the expiration of the invalid timer RIP would process any updates received by updating the route’s timers.

Hold-down timer (default value: 180 seconds)

When the invalid timer expires, the route automatically enters the hold-down phase. During hold-down, all updates regarding the route are disregarded -- it is assumed that the network may not have converged and that there may be bad routing information circulating in the network. The hold-down timer is started when the invalid timer expires. Thus, a route goes into hold-down state when the invalid timer expires. A route may also go into hold-down state when an update is received indicating that the route has become unreachable -- this is discussed further later in this section.

Flush timer (default value: 240 seconds)

The flush timer is set to when an update is received. When the flush timer expires, the route is removed from the routing table and the router is ready to receive an update with this route. Note that the flush timer overrides the hold-down timer.

Let’s consider [Figure 2-3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s03.html#iprouting-CHP-2-FIG-3 "Figure 2-3. Three routers connected using Ethernet segments"). Here is a snapshot of _A_’s routing table (when all entities are up):

![[Pasted image 20241110163935.png]]

Figure 2-3. Three routers connected using Ethernet segments

```
A>sh ip route
...

C       192.168.1.0 is directly connected, Ethernet1
     172.17.0.0/16 is subnetted, 6 subnets
C       172.17.1.9 is directly connected, Ethernet0
C       172.17.250.0 is directly connected, Ethernet1
C       172.17.251.0 is directly connected, Ethernet2
R       172.17.50.0 [120/1] via 172.17.250.2, 0:00:11, Ethernet1
R       172.17.100.0 [120/1] via 172.17.251.2, 0:00:19, Ethernet2
R       172.17.252.0 [120/1] via 172.17.250.2, 0:00:11, Ethernet1
                     [120/1] via 172.17.251.2, 0:00:19, Ethernet2
```

This table shows that 11 seconds ago _A_ received an update for `172.17.50.0` from `172.17.250.2` (_B_). The update and invalid timers for a route are reset (set to 0) every time a valid update is received for the route. At the moment this routing-table snapshot was taken, _A_’s invalid timer for `172.16.50.0` and _B_’s update timer for `172.16.50.0` would both be 11 seconds.

Let’s say that at this very time, _B_ was disconnected from its LAN attachment to _A_. _A_ would now stop receiving updates from _B_. 30 seconds after the cut, the routing table would look like this:

```
A>sh ip route
...

C       192.168.1.0 is directly connected, Ethernet1
     172.17.0.0/16 is subnetted, 6 subnets
C       172.17.1.9 is directly connected, Ethernet0
C       172.17.250.0 is directly connected, Serial0
C       172.17.251.0 is directly connected, Serial1
R       172.17.50.0 [120/1] via 172.17.250.2, 0:00:41, Serial0
R       172.17.100.0 [120/1] via 172.17.251.2, 0:00:19, Serial1
R       172.17.252.0 [120/1] via 172.17.250.2, 0:00:41, Serial0
                     [120/1] via 172.17.251.2, 0:00:19, Serial1
```

The invalid timer for `172.16.50.0` is now at 41 seconds. _A_ would still continue to forward traffic for `172.17.50.0` via _Ethernet0_. The assumption RIP makes is that an update was lost or damaged in transit from _B_ to _A_, even though the route is still good. This assumption holds good until the invalid timer expires (180 seconds or 6 update intervals from the last update). Before the invalid timer expires, _A_ will receive and process any updates received regarding `172.16.50.0`. Once the invalid timer expires, the route is placed in hold-down and subsequent updates about `172.16.0.0` are suppressed under the assumption that the route has gone bad and that bad routing information may be circulating in the network. The route will go into hold-down 180 seconds from the last update, or 169 seconds after the cut. At this time, the routing table would look like this:

```
A>sh ip route
...

C       192.168.1.0 is directly connected, Ethernet1
     172.17.0.0/16 is subnetted, 6 subnets
C       172.17.1.9 is directly connected, Ethernet0
C       172.17.250.0 is directly connected, Serial0
C       172.17.251.0 is directly connected, Serial1
R       172.17.50.0 is possibly down,
          routing via 172.17.250.2, Serial0
R       172.17.100.0 [120/1] via 172.17.251.2, 0:00:19, Serial1
R       172.17.252.0 [120/1] is possibly down,
          routing via 172.16.250.2, Ethernet1
                     [120/1] via 172.17.251.2, 0:00:19, Serial1
```

The route remains in hold-down until the hold-down timer expires or until the route is flushed, whichever happens first. Using default timers, the flush timer would go off first, 229 seconds after the cut. Router _A_ would then learn the route to `172.17.50.0` when the next update arrived from _C_, which could be between and 30 seconds after the route has been flushed, or 229 to 259 seconds from the cut.

The events just described are illustrated in [Figure 2-4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s03.html#iprouting-CHP-2-FIG-4 "Figure 2-4. Route convergence after a failure").

![[Pasted image 20241110164116.png]]

Figure 2-4. Route convergence after a failure

### Speeding Up Convergence

When a router detects that an interface is down, it immediately flushes all routes it knows via that interface. This speeds up convergence, avoiding the invalid, hold-down, and flush timers.

Can you now guess the reason why the case study used earlier (routers _A_, _B_, and _C_ connected via Ethernet segments) differs slightly from TraderMary’s network in New York, Chicago, and Ames?

We couldn’t illustrate the details of the invalid, hold-down, and flush timers in TraderMary’s network because if a serial link is detected in the down state, all routes that point through that interface are immediately flushed from the routing table. In our case study, we were able to pull _B_ off its Ethernet connection to _A_ while keeping _A_ up on all its interfaces.

#### Split horizon

Consider a simple network with two routers connected to each other (Figure 2-5).

![[Pasted image 20241110164154.png]]
Figure 2-5. Split horizon

Let’s say that router _A_ lost its connection to `172.18.1.0`, but before it could update _B_ about this change, _B_ sent _A_ its full routing table, including `172.18.1.0` at one hop. Router _A_ now assumes that _B_ has a connection to `172.18.1.0` at one hop, so _A_ installs a route to `172.18.1.0` at two hops via _B_. _A_’s next update to _B_ announces `172.18.1.0` at two hops, so _B_ adjusts its route `172.18.1.0` to three hops via _A_! This cycle continues until the route metric reaches 16, at which stage the route update is discarded.

Split horizon solves this problem by proposing a simple solution: when a router sends an update through an interface, it does not include in its update any routes that it learned via that interface. Using this rule, the only network that _A_ would send to _B_ in its update would be `172.18.1.0`, and the only network that _B_ would send to _A_ would be `172.18.2.0`. _B_ would never send `172.18.1.0` to _A_, so the previously described loop would be impossible.

#### Counting to infinity

Split horizon works well for two routers directly connected to each other. However, consider the following network (shown in [Figure 2-6](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s03.html#iprouting-CHP-2-FIG-6 "Figure 2-6. Counting to infinity")).

![[Pasted image 20241110164222.png]]
Figure 2-6. Counting to infinity

Let’s say that router _A_ stopped advertising network _X_ to its neighbors _B_ and _E_. Routers _B_, _D_, and _E_ will finally purge the route to _X_, but router _C_ may still advertise _X_ to _D_ (without violating split horizon). _D_, in turn, will advertise _X_ to _E_, and _E_ will advertise _X_ to _A_. Thus, the router (_C_) that did not purge _X_ from its table can propagate a bad route.

This problem is solved by equating a hop count of 16 to infinity and hence disregarding any advertisement for a route with this metric.

In [Figure 2-6](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s03.html#iprouting-CHP-2-FIG-6 "Figure 2-6. Counting to infinity"), when _B_ finally receives an advertisement for _X_ with a metric of 16, it will consider _X_ to be unreachable and will disregard the advertisement. The choice of 16 as infinity limits RIP networks to a maximum diameter of 15 hops between nodes. Note that the choice of 16 as infinity is a compromise between convergence time and network diameter -- if a higher number were chosen, the network would take longer to converge after a failure; if a lower number were chosen, the network would converge faster but the maximum possible diameter of a RIP network would be smaller.

#### Triggered updates

When a router detects a change in the metric for a route and sends an update to its neighbors right away (without waiting for its next update cycle), the update is referred to as a _triggered update_. The triggered update speeds convergence between two neighbors by as much as 30 seconds. A triggered update does not include the entire routing table, but only the route that has changed.

#### Poison reverse

When a router detects that a link is down, its next update for that route will contain a metric of 16. This is called _poisoning_ the route. Downstream routers that receive this update will immediately place the route in hold-down (without going through the invalid period).

Poison reverse and triggered updates can be combined. When a router detects that a link has been lost or the metric for a route has changed to 16, it will immediately issue a poison reverse with triggered update to all its neighbors.

Neighbors that receive unreachability information about a route via a poison reverse with triggered update will place the route in hold-down if their next hop is via the router issuing the poison reverse. The hold-down state ensures that bad information about the route (say from a neighbor that may have lost its copy of the triggered update or may have issued a regular update just before it received the triggered update) does not propagate in the network.

Triggered updates and hold-downs can handle the loss of a route, preventing bad routing information. Why, then, do we need the count-to-infinity limits? Triggered updates may be dropped, lost, or corrupted. Some routers may not ever receive the unreachability information and may inject a path for a route into the network even when that path has been lost. Count to infinity would take care of these situations.

#### Setting timers

The value of RIP timers on a Cisco router can be seen in the following example:

```
Chicago>sh ip protocol

Routing Protocol is "rip"
Sending updates every 30 seconds, next due in 24 seconds
Invalid after 90 seconds, hold down 90, flushed after 180
```

These timers could be modified to allow faster convergence. The following command:

```
timers basic 10 25 30 40
```

would send RIP updates every 10 seconds instead of every 30 seconds. The other three timers specify the invalid, hold-down, and flush timers, respectively. These timers can be configured as follows:

```
NewYork#config
NewYork-config#router rip
NewYork-config#timers basic 10 25 30 40
```

However, RIP timers should not be modified without a detailed understanding of how RIP works. Potential problems with decreasing the timer values are that updates will be issued more frequently and can cause congestion on low-bandwidth networks, and that congestion in the network is more likely to cause routes to go into hold-down; this, in turn, can cause route flapping.

---
#### Warning

Do not modify RIP timers unless absolutely necessary. If you modify RIP timers, make sure that all routers have the same timers.

---

If an interface on a router goes down, the router sends a RIP request out to the other, up interfaces. This speeds up convergence if any of the other neighbors can reach the destinations that were missed in the first request.

## 1.4 Subnet Masks

Looking closely at [Figure 2-2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s02.html#iprouting-CHP-2-FIG-2 "Figure 2-2. Format of RIP update packet"), we see that there is no field for subnet masks in RIP. Let’s say that router _SantaFe_ received an update with the following routes in the IP address field:

```
192.100.1.48
192.100.1.64
192.100.2.0
10.0.0.0
```

And let’s say that _SantaFe_ has the following configuration:

```
hostname SantaFe
!
interface Ethernet 0
ip address 192.100.1.17 255.255.255.240
!
interface Ethernet 1
ip address 192.100.1.33 255.255.255.240
!
router rip
network 192.100.1.0
network 192.100.2.0
```

How would the router associate subnet masks with these routes?

- If the router has an interface on a network number received in an update, it would associate the same mask with the update as it does with its own interface. Consequently, RIP does not permit Variable Length Subnet Masks (VLSM).
    
- If the router does not have an interface on the network number received in an update, it would assume a natural mask for the network number.
    

_SantaFe_’s routing table would look like this:

```
SantaFe>sh ip route
...
Gateway of last resort is not set

R       10.0.0.0 [120/1] via 192.100.1.18, 0:00:11, Ethernet0
R       192.100.2.0 [120/1] via 192.100.1.18, 0:00:11, Ethernet0
     192.100.1.0/16 is subnetted, 4 subnets
C       192.100.1.16 is directly connected, Ethernet0
C       192.100.1.32 is directly connected, Ethernet1
R       192.100.1.48 [120/1] via 192.100.1.18, 0:00:11, Ethernet0
R       192.100.1.64 [120/1] via 192.100.1.18, 0:00:11, Ethernet0
```

_SantaFe_ represents `192.100.1.48` and `192.100.1.64` with a 28-bit mask even though the subnet mask was not conveyed in the RIP update. _SantaFe_ was able to deduce the 28-bit mask because it has direct interfaces on `192.100.1.0` networks. This assumption is key to why RIP does not support VLSM.

_SantaFe_ represents `192.100.2.0` and `10.0.0.0` with their natural 24-bit and 8-bit masks, respectively, because it has no interfaces on those networks. [Chapter 5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch05.html "Chapter 5. Routing Information Protocol Version 2 (RIP-2)") covers RIP-2, an extension of RIP that supports VLSM.

## 1.5 Route Summarization

Consider the router _Phoenix_, which connects to _SantaFe_ and sends the RIP updates shown earlier:

```
192.100.1.48
192.100.1.64
192.100.2.0
10.0.0.0
```

_Phoenix_ may have been configured as follows (see [Figure 2-8](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s06.html#iprouting-CHP-2-FIG-8 "Figure 2-8. RIP routes to hosts"), later in [Chapter 2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02.html "Chapter 2. Routing Information Protocol (RIP)")):

```
hostname Phoenix
ip subnet-zero
!
interface Ethernet 0
ip address 192.100.1.18 255.255.255.240
!
interface Ethernet 1
ip address 192.100.1.49 255.255.255.240
!
interface Ethernet 2
ip address 192.100.1.65 255.255.255.240
!
interface Ethernet 3
ip address 192.100.2.1 255.255.255.240
!
interface Ethernet 4
ip address 192.100.2.17 255.255.255.240
!
interface Ethernet 5
ip address 10.1.0.1 255.255.0.0
!
interface Ethernet 6
ip address 10.2.0.1 255.255.0.0
!
router rip
network 192.100.1.0
network 192.100.2.0
network 10.0.0.0
```

_Phoenix_ did not send detailed routes for `192.100.2.0` or `10.0.0.0` when advertising to _SantaFe_ because _Phoenix_ summarized those routes. As I stated earlier, since _Phoenix_ did not have interfaces on those networks, it couldn’t have made sense of those routes anyway.

## 1.6 Default Route

A routing table need not contain all routes in the network to reach all destinations. This simplification is arrived at through the use of a _default route_ . When a router does not have an explicit route to a destination IP address, it looks to see if it has a default route in its routing table and, if so, forwards packets for this destination via the default route.

In RIP, the default route is represented as the IP address `0.0.0.0`. This is convenient because `0.0.0.0` cannot be confused with any Class A, B, or C IP address.

One situation in which default routes can be employed in an intranet is in a core network that has branch offices hanging off it ([Figure 2-7](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s06.html#iprouting-CHP-2-FIG-7 "Figure 2-7. Branch offices only need a default route")).

![[Pasted image 20241110164649.png]]
Figure 2-7. Branch offices only need a default route

Consider the topology of this figure. Since the branch offices have only one connection (to the core), all routes to the core network and to other branches can be replaced with a single default route pointing toward the core network. This implies that the size of the routing table in the branch offices is just the number of directly connected networks plus the default route.

So, router _Portland_ may be configured as follows:

```
hostname Portland
...
interface Ethernet 0
ip address 192.100.1.17 255.255.255.240
!
interface Serial 0
ip address 192.100.1.33 255.255.255.240
!
router rip
network 192.100.1.0
```

An examination of _Portland_’s routing table would show the following:

```
Portland>sh ip route
...
Gateway of last resort is not set

     192.100.1.0/28 is subnetted, 2 subnets
C       192.100.1.16 is directly connected, Ethernet0
C       192.100.1.32 is directly connected, Serial0
R    0.0.0.0 [120/1] via 192.199.1.34, 0:00:21, Serial0
```

The default route may be sourced from router _core1_ as follows:

```
hostname core1
...
interface Serial 0
ip address 192.100.1.34 255.255.255.240
!
router rip
network 192.100.1.0
!
ip route 0.0.0.0 0.0.0.0 null0
```

Note that the default route `0.0.0.0` is automatically carried by RIP -- it is not listed in a network number statement under _router rip_.

The advantage of using a default in place of hundreds or thousands of more specific routes is obvious -- network bandwidth and router CPU are not tied up in routing updates. The disadvantage of using a default is that packets for destinations that are down or not even defined in the network are still forwarded to the core network.

Default routes are tremendously useful in Internet connectivity -- where all (thousands and thousands of ) Internet routes may be represented by a single default route.

Yet another use of default routes is in maintaining reachability between a routing domain running RIP and another routing domain with VLSM. Since VLSM cannot be imported into RIP, a default route pointing to the second domain may be defined in the RIP network.

### Routes to hosts

Some host machines listen to RIP updates in “quiet” or “silent” mode ([Figure 2-8](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s06.html#iprouting-CHP-2-FIG-8 "Figure 2-8. RIP routes to hosts")). These hosts do not respond to requests for RIP routes or issue regular RIP updates. Listening to RIP provides redundancy to the hosts in a scenario in which multiple routers are connected to a segment. If the routers have similar routing tables, it may make sense to send only the default route (`0.0.0.0`) to hosts.

![[Pasted image 20241110164753.png]]
Figure 2-8. RIP routes to hosts

## 1.7 Fine-Tuning RIP

We saw in the section on RIP metrics that the preferred path between _NewYork_ and _Ames_ would be the two-hop path via _Chicago_ rather than the one-hop 56-kbps path that RIP selects. The RIP metrics can be manipulated to disfavor the one-hop path through the use of offset lists:

```
hostname NewYork
...
router rip
network 172.16.0.0
offset-list 10 in 2 serial1
...
access-list 10 permit 172.16.100.0 0.0.0.0

hostname Chicago
...
router rip
network 172.16.0.0

Ames#config terminal
router rip
network 172.16.0.0
offset-list 20 in 2 serial1
...
access-list 20 permit 172.16.1.0 0.0.0.0
```

_NewYork_ adds 2 to the metric for the routes specified in access list 10 when learned via _serial1_, and _Ames_ adds 2 to the metric for the routes specified in access list 20 when learned via _serial1_. The direct route over the 56-kbps link thus has a metric of 3, and the route via _Chicago_ has a metric of 2. The new routing tables look like this:

```
NewYork>sh ip route
...
Gateway of last resort is not set

C       192.168.1.0 is directly connected, Ethernet1
     172.16.0.0/16 is subnetted, 6 subnets
C       172.16.1.0 is directly connected, Ethernet0
C       172.16.250.0 is directly connected, Serial0
C       172.16.251.0 is directly connected, Serial1
R       172.16.50.0 [120/1] via 172.16.250.2, 0:00:11, Serial0
R       172.16.100.0 **`[120/2]`** via 172.16.250.2, 0:00:19, Serial0
R       172.16.252.0 [120/1] via 172.16.250.2, 0:00:11, Serial0
                     [120/1] via 172.16.251.2, 0:00:19, Serial1

Ames>sh ip route
...
Gateway of last resort is not set

     172.16.0.0/16 is subnetted, 6 subnets
C       172.16.100.0 is directly connected, Ethernet0
C       172.16.252.0 is directly connected, Serial0
C       172.16.251.0 is directly connected, Serial1
R       172.16.50.0 [120/1] via 172.16.252.1, 0:00:21, Serial0
R       172.16.1.0 **`[120/2]`** via 172.16.251.1, 0:00:09, Serial1
R       172.16.250.0 [120/1] via 172.16.252.1, 0:00:21, Serial0
                     [120/1] via 172.16.251.1, 0:00:09, Serial1
```

The syntax for offset lists is as follows:

```
offset-list {_`access-list`_} {in | out}_`offset`_ [_`type number`_]
```

The offset list specifies the offset to add to the RIP metric on routes of interface _type_ (Ethernet, serial, etc.) and _number_ (interface number) that are being learned (_in_) or advertised (_out_).

An offset list can also be applied to default routes. Thus, in [Figure 2-7](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s06.html#iprouting-CHP-2-FIG-7 "Figure 2-7. Branch offices only need a default route"), let’s consider the scenario where _Portland_ is given a second connection to a backup router _core2_. _core2_ may originate a default with a higher metric:

```
hostname core2
...
interface Serial 0
ip address 192.100.2.34 255.255.255.240
!
router rip
network 192.100.2.0
offset-list 30 out 3 serial0
!
ip route 0.0.0.0 0.0.0.0 null0
!
access-list 30 permit 0.0.0.0 0.0.0.0
```

_Portland_ would prefer the default via _core1_ because the metric from _core1_ would be lower by 3. _Portland_ would use the default from _core2_ if _core1_ or the link to _core1_ went down.

## 1.8 Summing Up

RIP is a relatively simple protocol, easy to configure and very reliable. The robustness of RIP is evident from the fact that various implementations of RIP differ in details and yet work well together. A standard for RIP wasn’t put forth until 1988 (by Charles Hedrick, in RFC 1058). Small, homogeneous networks are a good match for RIP. However, as networks grow, other routing protocols may look more attractive for several reasons:

- The RIP metric does not account for link bandwidth or delay.
    
- The exchange of full routing updates every 30 seconds does not scale for large networks -- the overhead of generating and processing all routes can be high.
    
- RIP convergence times can be too long.
    
- Subnet mask information is not exchanged in RIP updates, so Variable Length Subnet Masks are not supported.
    
- The RIP metric restricts the network diameter to 15 hops.

# Chapter 2. Interior Gateway Routing Protocol (IGRP)

The second Distance Vector protocol that we will examine is the Interior Gateway Routing Protocol, or IGRP. IGRP and RIP are close cousins: both are based on the Bellman-Ford Distance Vector (DV) algorithms. DV algorithms propagate routing information from neighbor to neighbor; if a router receives the same route from multiple neighbors, it chooses the route with the lowest metric. All DV protocols need robust strategies to cope with _bad_ routing information. Bad routes can linger in a network when information about the loss of a route does not reach some router (for instance, because of the loss of a route update packet), which then inserts the bad route back into the network. IGRP uses the same convergence strategies as RIP: triggered updates, route hold-downs, split horizon, and poison reverse.

IGRP has been widely deployed in small to mid-sized networks because it can be configured with the same ease as RIP, but its metric represents bandwidth and delay, in addition to hop count. The ability to discriminate between paths based on bandwidth and delay is a major improvement over RIP.

IGRP is a Cisco proprietary protocol; other router vendors do not support IGRP. Keep this in mind if you are planning a multivendor router environment.

The following section gets us started with configuring IGRP.

## 2.1 Getting IGRP Running

TraderMary’s network, shown in Figure 3-1

![[Pasted image 20241110170001.png]]
Figure 3-1. TraderMary’s network

Like RIP, IGRP is a distributed protocol that needs to be configured on every router in the network:

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
description New York to Chicago link
ip address 172.16.250.1 255.255.255.0
!
interface Serial1
description New York to Ames link
ip address 172.16.251.1 255.255.255.0
...
router igrp 10
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

router igrp 10
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

router igrp 10
network 172.16.0.0
```

The syntax of the IGRP command is:

```
router igrp {_`process-id`_ | _`autonomous-system-number`_}
```

in global configuration mode. The networks that will be participating in the IGRP process are then listed:

```
network 172.16.0.0
```

What does it mean to list the network numbers participating in IGRP?

1. _NewYork_ will include directly connected `172.16.0.0` subnets in its updates to neighboring routers. For example, `172.16.1.0` will now be included in updates to the routers _Chicago_ and _Ames_.
    
2. _NewYork_ will receive and process IGRP updates on its `172.16.0.0` interfaces from other routers running IGRP 10. For example, _NewYork_ will receive IGRP updates from _Chicago_ and _Ames_.
    
3. By exclusion, network `192.168.1.0`, connected to _NewYork,_ will not be advertised to _Chicago_ or _Ames_, and _NewYork_ will not process any IGRP updates received on _Ethernet0_ (if there is another router on that segment).
    

Next, let’s verify that all the routers are seeing all the `172.16.0.0` subnets. Here is _NewYork_’s routing table:

```
NewYork#show ip route
Codes: C - connected, S - static, I - IGRP, R - RIP, M - mobile, B - BGP
       D - EIGRP, EX - EIGRP external, O - OSPF, IA - OSPF inter area
       N1 - OSPF NSSA external type 1, N2 - OSPF NSSA external type 2
       E1 - OSPF external type 1, E2 - OSPF external type 2, E - EGP
       i - IS-IS, L1 - IS-IS level-1, L2 - IS-IS level-2, * - candidate default
       
Gateway of last resort is not set

     172.16.0.0/24 is subnetted, 6 subnets
I       172.16.252.0 [100/10476] via 172.16.251.2, 00:00:26, Serial1
                     [100/10476] via 172.16.250.2, 00:00:37, Serial0
C       172.16.250.0 is directly connected, Serial0
C       172.16.251.0 is directly connected, Serial1
I       172.16.50.0 [100/8576] via 172.16.250.2, 00:00:37, Serial0
C       172.16.1.0 is directly connected, Ethernet0
I       172.16.100.0 [100/8576] via 172.16.251.2, 00:00:26, Serial1
C    192.168.1.0/24 is directly connected, Ethernet1
```

Here is _Chicago_’s table:

```
Chicago#sh ip route
...
Gateway of last resort is not set

     172.16.0.0/24 is subnetted, 6 subnets
C       172.16.252.0 is directly connected, Serial1
C       172.16.250.0 is directly connected, Serial0
I       172.16.251.0 [100/10476] via 172.16.250.1, 00:01:22, Serial0
                     [100/10476] via 172.16.252.2, 00:00:17, Serial1
C       172.16.50.0 is directly connected, Ethernet0
I       172.16.1.0 [100/8576] via 172.16.250.1, 00:01:22, Serial0
I       172.16.100.0 [100/8576] via 172.16.252.2, 00:00:17, Serial1
```

And here is _Ames_’s table:

```
Ames#sh ip route
...
Gateway of last resort is not set

     172.16.0.0/24 is subnetted, 6 subnets
C       172.16.252.0 is directly connected, Serial0
I       172.16.250.0 [100/10476] via 172.16.251.1, 00:01:11, Serial1
                     [100/10476] via 172.16.252.1, 00:00:21, Serial0
C       172.16.251.0 is directly connected, Serial1
I       172.16.50.0 [100/8576] via 172.16.252.1, 00:00:21, Serial0
I       172.16.1.0 [100/8576] via 172.16.251.1, 00:01:11, Serial1
C       172.16.100.0 is directly connected, Ethernet0
```

The IGRP-derived routes in these tables are labeled with an “I” in the left margin. The first line in each router’s table contains summary information:

```
172.16.0.0/**`24`** is subnetted, **`6`** subnets
```

Note that all three routers show the same summary information -- _NewYork_, _Chicago_, and _Ames_ show all six subnets.

Note also that network `192.168.1.0`, defined on _NewYork_ interface _Ethernet1_, did not appear in the routing tables of _Chicago_ and _Ames_. To be propagated, `192.168.1.0` would have to be defined in a network statement under the IGRP configuration on _NewYork_:

```
hostname NewYork
...
router igrp 10
network 172.16.0.0
network 192.168.1.0
```

Getting IGRP started is fairly straightforward. However, if you compare the routing tables in this section to those in the previous chapter on RIP, there is no difference in the next-hop information. More importantly, the route from _NewYork_ to network `172.16.100.0` is still over the direct 56-kbps path rather than the two-hop T-1 path. The two-hop T-1 path is better than the one-hop 56-kbps link. As an example, take a 512-byte packet; it would take 73 ms to copy this packet over a 56-kbits/s link versus 5 ms over two T-1 links. Our expectation is that IGRP should install this two-hop T-1 path, since IGRP has been touted for its metric that includes link bandwidth and delay. Section 2.2.2 explains why IGRP installs the slower path. Section 2.2.2.6 leads us through the configuration changes required to make IGRP install the faster path.

A key difference in this configuration is that, unlike in RIP, each IGRP process is identified by an autonomous system (AS) number. AS numbers are described in detail in the next section.

## 2.2 How IGRP Works

Since IGRP is such a close cousin of RIP, we will not repeat the details of how DV algorithms work, how updates are sent, and how route convergence is achieved. However, because IGRP employs a much more comprehensive metric, I’ll discuss the IGRP metric in detail. I’ll begin this discussion with AS numbers.

### 2.2.1 IGRP Autonomous System Number

Each IGRP process requires an autonomous system number:

router igrp _`autonomous-system-number`_

The AS number allows the network administrator to define routing domains; routers within a domain exchange IGRP routing updates with each other but not with routers in different domains. Note that in the context of IGRP the terms “autonomous system number” and “process ID” are often used interchangeably. Since the IGRP autonomous system number is not advertised to other domains, network engineers often cook up arbitrary process IDs for their IGRP domains.

Let’s say that TraderMary created a subsidiary in Africa and that the new topology is as shown in [Figure 3-2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-FIG-2 "Figure 3-2. TraderMary’s U.S. and African networks").

![[Pasted image 20241110170821.png]]
Figure 3-2. TraderMary’s U.S. and African networks

Note that IGRP is running in the U.S. and Africa with AS numbers of 10 and 20, respectively. The U.S. routers now exchange IGRP routes with each other, as before, and the routers _Nairobi_ and _Casablanca_ exchange IGRP updates with each other. IGRP updates are processed only between routers running the same AS number, so _NewYork_ and _Nairobi_ do not exchange IGRP updates with each other. We will see this in more detail later, when we look at the format of an IGRP update.

The advantage of creating small domains with unique AS numbers is that a routing problem in one domain is not likely to ripple into another domain running a different AS number. So, for example, let’s say that a network engineer in Africa configured network `172.16.50.0` on _Casablanca_ (`172.16.50.0` already exists on _Chicago_). The U.S. network would not be disrupted because of this duplicate address. In another situation, an IGRP bug in IOS on _Chicago_ could disrupt routing in the U.S., but _Nairobi_ and _Casablanca_ would not be affected by this problem in the AS 10.

The problem with creating too many small domains running different IGRP AS numbers is that sooner or later the domains will need to exchange routes with each other. The office in New York would need to send files to Nairobi. This could be accomplished by adding static routes on _NewYork_ (to `172.16.150.0`) and _Nairobi_ (to `172.16.1.0`). However, static routes can be cumbersome to install and administer and do not offer the redundancy of dynamic routing protocols. Dynamic distribution of routes between routing domains is discussed in [Chapter 8](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch08.html "Chapter 8. Administrative Controls").

In the meantime, all I will say is to use good judgment when breaking networks into autonomous systems. Making a routing domain too small will require extensive redistributions or the creation of static entries. Making a routing domain too big exposes the network to failures of the type just described.

The boundary between domains is often geographic or organizational.

### 2.2.2 IGRP Metric

The RIP metric was designed for small, homogenous networks. Paths were selected based on the number of hops to a destination; the lowest hop-count path was installed in the routing table. IGRP is designed for more complex networks. Cisco’s implementation of IGRP allows the network engineer to customize the metric based on bandwidth, delay, reliability, load, and MTU. In order to compare metrics between paths and select the least-cost path, IGRP converts bandwidth, delay, reliability, delay, and MTU into a scalar quantity -- a _composite_ metric that expresses the desirability of a path. Just as in the case of RIP, a path with a lower composite metric is preferred to a path with a higher composite metric.

The computation of the IGRP composite metric is user-configurable; i.e., the network administrator can specify parameters in the _formula_ used to convert bandwidth, delay, reliability, load, and MTU into a scalar quantity.

The following sections define bandwidth, delay, reliability, load, and MTU. We will then see how these variables can be used to compute the composite metric for a path.

#### 2.2.2.1 Interface bandwidth, delay, reliability, load, and MTU

The IGRP metric for a path is derived from the bandwidth, delay, reliability, load, and MTU values of every media in the path to the destination network.

The bandwidth, delay, reliability, load, and MTU values at any interface to a media can be seen as output of the **show interface** command:

```
router1#sh interface ethernet 0
Ethernet0 is up, line protocol is up 
  Hardware is AmdP2, address is 0010.7bcf.e340 (bia 0010.7bcf.e340)
  Description: Lab Test
  Internet address is 1.13.96.1/16
  **`MTU 1500 bytes, BW 10000 Kbit, DLY 1000 usec, rely 255/255, load 1/255`**
  Encapsulation ARPA, loopback not set, keepalive set (10 sec)
  ARP type: ARPA, ARP Timeout 04:00:00
...
```

The bandwidth, delay, reliability, load, and MTU for a media are defined as follows:

Bandwidth

The bandwidth for a link represents how fast the physical media can transmit bits onto a wire. Thus, an HSSI link transmits approximately 45,000 kbits every second, Ethernet runs at 10,000 kbps, a T-1 link transmits 1,544 kbits every second, and a 56-kbps link transmits 56 kbits every second.

_Ethernet0_ on _router1_ is configured with a bandwidth of 10,000 kbps.

Delay

The delay for a link represents the time to traverse the link in an unloaded network and includes the propagation time for the media. Ethernet has a delay value of 1 ms; a satellite link has a delay value in the neighborhood of 1 second.

_Ethernet0_ on _router1_ is configured with a delay of 1,000 ms.

Reliability

Link reliability is dynamically measured by the router and is expressed as a numeral between 1 and 255. A reliability of 255 indicates a 100% reliable link.

_Ethernet0_ on _router1_ is 100% reliable.

Load

Link utilization is dynamically measured by the router and is expressed as a numeral between 1 and 255. A load of 255 indicates 100% utilization.

_Ethernet0_ on _router1_ has a load of 1/255.

MTU

The MTU, or Maximum Transmission Unit, represents the largest frame size the link can handle.

_Ethernet0_ on _router1_ has an MTU size of 1,500 bytes.

The MTU, bandwidth, and delay values are static parameters that Cisco routers derive from the media type. [Table 3-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-TABLE-1 "Table 3-1. Default bandwidth and delay values") shows some common values for bandwidth and delay. These default values can be modified using the commands shown in the next section.

Table 3-1. Default bandwidth and delay values

| Media type                 | Default bandwidth | Default delay |
| -------------------------- | ----------------- | ------------- |
| Ethernet                   | 10 Mbps           | 1,000 ms      |
| Fast Ethernet              | 100 Mbps          | 100 ms        |
| FDDI                       | 100 Mbps          | 100 ms        |
| T-1 (serial interface)     | 1,544 kbps        | 20,000 ms     |
| 56 kbps (serial interface) | 1,544 kbps        | 20,000 ms     |
| HSSI                       | 45,045 kbps       | 20,000 ms     |
|                            |                   |               |

 All serial interfaces on Cisco routers are configured with the same _default_ bandwidth (1,544 kbits/s) and delay (20,000 ms) parameters.
 
The reliability and load values are dynamically computed by the router as five-minute exponentially weighted averages.

#### 2.2.2.2 Modifying interface bandwidth, delay, and MTU

The default bandwidth and delay values may be overridden by the following interface commands:

```
bandwidth kilobits
delay tens-of-microseconds
```

So, the following commands will define a bandwidth of 56,000 bps and a delay of 10,000 ms on interface _Serial0_:

```
interface Serial0
bandwidth 56
delay 1000
```

These settings affect only IGRP routing parameters. The actual physical characteristics of the interface -- the clock-rate on the wire and the media delay -- have no relationship to the bandwidth or delay values configured as in this example or seen as output of the **show interface** command. Thus, interface _Serial0_ in the previous example may actually be clocking data at 128,000 bps, a rate that will be governed by the configuration of the modem or the CSU/DSU attached to _Serial0_. Note that, by default, Cisco sets the bandwidth and delay on all serial interfaces to be 1,544 kbps and 20,000 ms, respectively (see [Table 3-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-TABLE-1 "Table 3-1. Default bandwidth and delay values")).

Note that delay on an interface is specified in _tens of microseconds._ Thus:

```
delay 1000
```

describes a delay of 10,000 ms.

The MTU on an interface can be modified using the following command:

```
mtu bytes
```

However, the MTU size has no bearing on IGRP route selection. The MTU size should not be modified to affect routing behavior. The default MTU values represent the maximum allowed for the media type; lowering the MTU size can impair performance by causing needless fragmentation of IP datagrams.

Later in this chapter we will see how modifications to the bandwidth and delay parameters on an interface can affect route selection.

#### 2.2.2.3 IGRP routing update

IGRP updates are directly encapsulated in IP with the protocol field (in the IP header) set to 9. The format of an IGRP packet is shown in [Figure 3-3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-FIG-3 "Figure 3-3. Format of an IGRP update packet").

![[Pasted image 20241110171232.png]]
Figure 3-3. Format of an IGRP update packet

Just like RIP, IGRP allows a station to request routes. This allows a router that has just booted up to request the routing table from its neighbors instead of waiting for the next cycle of updates, which could be as much as 90 seconds later for IGRP.

The destination IP address in IGRP updates is `255.255.255.255`. The source IP address is the IP address of the interface from which the update is issued.

Each update packet contains three types of routes:

_Interior_ routes
	Contain subnet information for the major network number associated with the address of the interface to which the update is being sent. If the IGRP update is being sent on a broadcast network, the internal routes are subnet numbers from the same major network number that is configured in the broadcast media.

_System_ routes
	Contain major network numbers that may have been summarized when a network-number boundary was crossed.

_Exterior_ routes
	Represent candidates for the default route. Unlike RIP, which uses `0.0.0.0` to represent the default, IGRP uses specific network numbers as candidates for the default by tagging the routes as exterior.

Interior, system, and exterior routes appear in order in each update packet. The count of interior, system, and exterior routes identifies the route type for each route entry.

Note that the IGRP update has only three octets for the destination network-number field, whereas IP addresses are four octets in length. IGRP extracts the four-octet IP address using the heuristic shown in [Table 3-2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-TABLE-2 "Table 3-2. Deriving the four-octet IP destination address from the three-octet destination field").

Table 3-2. Deriving the four-octet IP destination address from the three-octet destination field

|Route type|Heuristic to derive four-octet IP destination address|
|---|---|
|Interior route|The first octet is derived from the IP address of the interface that received the update; the last three octets are derived from the IGRP update.|
|System route|The route is assumed to have been summarized. The last octet of the IP destination address is 0.|
|Exterior route (default route)|The route is assumed to have been summarized. The last octet of the IP destination address is 0.|

Just like RIP, IGRP updates do not contain subnet mask information. This classifies both RIP and IGRP as _classful_ routing protocols. Subnet mask information for routes received in IGRP updates is derived using the same rules as in RIP.

When an update is received for a route, it contains the bandwidth, delay, reliability, load, and MTU values for the path to the destination network via the source of the update. I already defined bandwidth, delay, reliability, load, and MTU for an interface. Now let’s define these parameters again for a path.

#### 2.2.2.4 Path bandwidth, delay, reliability, load, and MTU

The following list defines bandwidth, delay, reliability, load, and MTU for a path:

Bandwidth
	The bandwidth for a path is the minimum bandwidth in the path to the destination network. Compare a network to a sequence of pipes for the transmission of a fluid; the slowest pipe (or the thinnest pipe) will dictate the rate of flow of the fluid. Thus, if a path to a network is through an Ethernet segment, a T-1 line, and another Ethernet segment, the path bandwidth will be 1,544 kbps (see [Table 3-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-TABLE-1 "Table 3-1. Default bandwidth and delay values")).

Delay
	The delay for a path is the sum of all delay values on the path to the destination network. The IGRP unit of delay is in tens of microseconds. A path through a network via an Ethernet segment, a T-1 line, and another Ethernet segment will have a path delay of 22,000 ms or 2,200 IGRP delay units (see [Table 3-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-TABLE-1 "Table 3-1. Default bandwidth and delay values")).

The IGRP update packet has three octets to represent delay (in units of tens of microseconds). The largest value of delay that can be represented is 224 x 10 ms, which is roughly 167.7 seconds. 167.7 seconds is thus the maximum possible delay value for an IGRP network. All ones in the delay field are also used to indicate that the network indicated is _unreachable_.

Reliability
	The reliability for a path is the reliability of the least reliable link in the path.

Load
	The load for a path is the load on the most heavily loaded link in the path.

MTU
	The MTU represents the smallest MTU along the path. MTU is currently not used in computing the metric.

Note that, in addition to these parameters, the update packet includes the hop count to the destination. The default maximum hop count for IGRP is 100. This default can be modified with the command:

```
metric maximum-hops hops
```

The maximum value for _hops_ is 255. A network with a diameter over 100 is very large indeed, especially for a network running IGRP. Do not expect to modify the maximum hop count for IGRP, even if you are working for an interstellar ISP. Large networks will usually require routing features that do not exist in IGRP.

The bandwidth, delay, reliability, load, and MTU values for the path selected by a router can be seen as output of the **show ip route** command:

```
NewYork#show ip route 172.16.100.0
Routing entry for 172.16.100.0 255.255.255.0
  Known via "igrp 10", distance 100, metric 8576
  Redistributing via igrp 10
  Advertised by igrp 10 (self originated)
  Last update from 172.16.251.2 on Serial1, 00:00:29 ago
  Routing Descriptor Blocks:
  * 172.16.251.2, from 172.16.251.2, 00:00:29 ago, via Serial1
      Route metric is 8576, traffic share count is 1
      Total delay is 21000 microseconds, minimum bandwidth is 1544 Kbit
      Reliability 255/255, minimum MTU 1500 bytes
      Loading 1/255, Hops 2
```
#### 2.2.2.5 IGRP composite metric

The path metric of bandwidth, delay, reliability, load, and MTU needs to be expressed as a composite metric for you to be able to compare paths. The default behavior of Cisco routers considers only bandwidth and delay in computing the composite metric (the parameters reliability, load, and MTU are ignored):

_Metric = BandW + Delay_

_BandW_ is computed by taking the smallest bandwidth (expressed in kbits/s) from all outgoing interfaces to the destination (including the destination) and dividing 10,000,000 by this number (the smallest bandwidth). For example, if the path from a router to a destination Ethernet segment is via a T-1 link, then:

_BandW =_ 10,000,000/1,544 = 6,476

_Delay_ is computed by adding the delays from all outgoing interfaces to the destination (including the delay on the interface connecting to the destination network) and dividing by 10:

_Delay_ = (20,000 + 1,000)/10 = 2,100

And then the composite metric for the path to the Ethernet segment would be:

_Metric = BandW + Delay_ = 1,000 + 2,100 = 3,100

Let’s now go back to TraderMary’s network to see why router _NewYork_ selected the direct 56-kbps link to route to 172.16.100.0 and not the two-hop T-1 path via _Chicago_:

```
NewYork>sh ip route
...
I       172.16.100.0 [100/**`8576`**] via 172.16.251.2, 0:00:31, Serial0
...
```

The values of the IGRP metrics for these paths can be seen here:

```
Ames#sh interface Ethernet 0
Ethernet0 is up, line protocol is up 
  Hardware is Lance, address is 00e0.b056.1b8e (bia 00e0.b056.1b8e)
  Description: Lab Test
  Internet address is 172.16.100.1/24
  **`MTU 1500 bytes, BW 10000 Kbit, DLY 1000 usec, rely 255/255, load 1/255`**
  Encapsulation ARPA, loopback not set, keepalive set (10 sec)
...

NewYork# show interfaces serial 0
Serial 0 is up, line protocol is up
Hardware is MCI Serial
Internet address is 172.16.250.1, subnet mask is 255.255.255.0
MTU 1500 bytes, BW 1544 Kbit, DLY 20000 usec, rely 255/255, load 1/255
Encapsulation HDLC, loopback not set, keepalive set (10 sec)
...
```

There are two paths to consider:

1. _NewYork_ → _Ames_ → `172.16.100.0`.
    Bandwidth values in the path: (serial link) 1,544 kbits/s, (Ethernet segment) 10,000 kbits/s
    Delay values in the path: (serial link) 2,000, (Ethernet segment) 100
    Smallest bandwidth in the path: 1,544
    _BandW_ = 10,000,000/1,544 = 6,476
    _Delay_ = 2,000 + 100 = 2,100
    _Metric_ = _BandW_ + _Delay_ = 8,576
    
2. _NewYork_ → _Chicago_ → _Ames_ to `172.16.100.0`.
    Bandwidth values in the path: (serial link) 1,544 kbits/s, (serial link) 1,544 kbits/s, (Ethernet segment) 10,000 kbits/s
    Delay values in the path: (serial link) 2,000, (serial link) 2,000, (Ethernet segment) 100
    Smallest bandwidth in the path: 1,544
    _BandW_ = 10,000,000/56 = 6,476
    _Delay_ = 2,000 + 2,000 + 100 = 4,100
    _Metric_ = _BandW_ + _Delay_ = 10,576
    

_NewYork_ will prefer to route via the first path because the metric is smaller. Why does _NewYork_ use a bandwidth of 1,544 for the 56-kbps link to _Ames_? Go back to [Table 3-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-TABLE-1 "Table 3-1. Default bandwidth and delay values") and you will see that the default bandwidth and delay values of 1,544 kbps and 20,000 ms apply to all serial interfaces, regardless of the speed of the modem device attached to the router port.

The IGRP metric can be customized to use reliability and load with the following formula (Equation 1):

_Metric_ _= k1_ x _BandW + k2_ x _BandW/(256 - load) + k3_ x _Delay_

where the default values of the constants are k1 = k3 = 1 and k2 = k4 = k5 = 0.

If k5 is not equal to zero, an additional operation is done:

_Metric = Metric_ x [_k5_/(_reliability + k4_)]

The constants k1, k2, k3, k4, and k5 can be modified with the command:

```
metric weights tos k1 k2 k3 k4 k5
```

where _tos_ identifies the type of service and must be set to zero (because only one type of service has been defined).

Plugging the default values of k1, k2, k3, k4, and k5 into Equation 1 yields:

_Metric = BandW + Delay_

which we saw earlier.

To make the metric sensitive to the network load (in addition to bandwidth and delay), set k1 = k2 = k3 = 1 and k4 = k5 = 0. This yields:

_Metric = BandW + BandW_/(_256 - load_) _+ Delay_

The problem with using load in the metric computation is that it can make a route unstable. For example, a router may select a path through router _P_ as its next hop to reach a destination. When the load on the path through _P_ rises, in a few minutes (the value of load is computed as a five-minute exponentially weighted average) the metric for the path through _P_ may become larger than the metric for an alternative path through router _Q_. The traffic then shifts to _Q_; this causes the load to increase on the path through _Q_ and the path through _P_ becomes more attractive. Thus, setting k2 = 1 can make a route unstable and cause traffic to bounce between two paths. Further, abrupt changes in metric cause flash updates; the route may also go into hold-down.

Instead of selecting the best path based on load, you may consider load balancing over several paths. Load balancing occurs automatically over equal-cost paths. If two or more paths have slightly different metrics, you may consider modifying the bandwidth and delay parameters to make the metrics equal and to utilize all the paths. See the example on modifying bandwidth and delay parameters in the next section.

To make the metric sensitive to network reliability (in addition to bandwidth and delay), set k1 = k3 = k5 =1 and k2 = k4 = 0. In the event of link errors, this will cause the metric on the path to increase, and IGRP will select an alternative path when the metric has worsened enough. A typical action in today’s networks is to turn a line down until the transmission problem is resolved, not to base routing decisions on how badly the line is running.

#### Warning

Cisco strongly recommends _not_ modifying the k1, k2, k3, k4, and k5 values for IGRP.

#### 2.2.2.6 Modifying IGRP metrics

TraderMary’s network was still using the 56-kbps path between _NewYork_ and _Ames_, even when IGRP was running on the routers (refer to [Section 3.1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03.html#iprouting-CHP-3-SECT-1 "Getting IGRP Running")). Why is it that _NewYork_ and _Ames_ did not pick up the lower bandwidth for the 56-kbps link?

[Table 3-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-TABLE-1 "Table 3-1. Default bandwidth and delay values") contains the key to our question. All serial interfaces on a Cisco router are configured with the same bandwidth (1,544 kbps) and delay (20,000 ms) values. Thus, IGRP sees the 56-kbps line with the same bandwidth and delay parameters as a T-1 line.

In order to utilize the 56-kbps link only as backup, we need to modify TraderMary’s network as follows:

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
description New York to Chicago link
ip address 172.16.250.1 255.255.255.0
!
interface Serial1
description New York to Ames link
**`bandwidth 56`**
ip address 172.16.251.1 255.255.255.0
...
router igrp 10
network 172.16.0.0


hostname Chicago
...
interface Ethernet0
ip address 172.16.50.1 255.255.255.0
!
interface Serial0
description Chicago to New York link
ip address 172.16.250.2 255.255.255.0
!
interface Serial1
description Chicago to Ames link
ip address 172.16.252.1 255.255.255.0
...

router igrp 10
network 172.16.0.0


hostname Ames
...
interface Ethernet0
ip address 172.16.100.1 255.255.255.0
!
interface Serial0
description Ames to Chicago link
ip address 172.16.252.2 255.255.255.0
!
interface Serial1
description Ames to New York link
bandwidth 56
ip address 172.16.251.2 255.255.255.0
...

router igrp 10
network 172.16.0.0
```

The new routing tables look like this:

```
NewYork#show ip route
...
Gateway of last resort is 0.0.0.0 to network 0.0.0.0

     172.16.0.0/24 is subnetted, 6 subnets
I       172.16.252.0 [100/10476] via 172.16.250.2, 00:00:43, Serial0
C       172.16.250.0 is directly connected, Serial0
C       172.16.251.0 is directly connected, Serial1
I       172.16.50.0 [100/8576] via 172.16.250.2, 00:00:43, Serial0
C       172.16.1.0 is directly connected, Ethernet0
I       172.16.100.0 [100/10576] via 172.16.250.2, 00:00:43, Serial0
C    192.168.1.0/24 is directly connected, Ethernet1


Chicago#sh ip route
...
Gateway of last resort is not set

     172.16.0.0/24 is subnetted, 6 subnets
C       172.16.252.0 is directly connected, Serial1
C       172.16.250.0 is directly connected, Serial0
I       172.16.251.0 [100/182571] via 172.16.250.1, 00:00:01, Serial0
                     [100/182571] via 172.16.252.2, 00:01:01, Serial1
C       172.16.50.0 is directly connected, Ethernet0
I       172.16.1.0 [100/8576] via 172.16.250.1, 00:00:01, Serial0
I       172.16.100.0 [100/8576] via 172.16.252.2, 00:01:01, Serial1


Ames#sh ip route
...
Gateway of last resort is not set

     172.16.0.0/24 is subnetted, 6 subnets
C       172.16.252.0 is directly connected, Serial0
I       172.16.250.0 [100/10476] via 172.16.252.1, 00:00:24, Serial0
C       172.16.251.0 is directly connected, Serial1
I       172.16.50.0 [100/8576] via 172.16.252.1, 00:00:24, Serial0
I       172.16.1.0 [100/10576] via 172.16.252.1, 00:00:24, Serial0
C       172.16.100.0 is directly connected, Ethernet0
```

Let’s now go back to TraderMary’s network and corroborate the metric values seen for `172.16.100.0` in router _NewYork’s_ routing table. The following calculations show TraderMary’s network as in [Figure 3-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03.html#iprouting-CHP-3-FIG-1 "Figure 3-1. TraderMary’s network") but with IGRP bandwidth and delay values for each interface. There are two paths to consider:

1. _NewYork_ → _Ames_ → `172.16.100.0`.
    Bandwidth values in the path: (serial link) 56 kbits/s, (Ethernet segment) 10,000 kbits/s
    Smallest bandwidth in the path: 56
    _BandW_ = 10,000,000/56 = 178,571
    _Delay_ = 2,000 + 100 = 2100
    _Metric_ = _BandW_ + _Delay_ = 180,671
    
2. _NewYork_ → _Chicago_ → _Ames_ → `172.16.100.0`
    Bandwidth values in the path: (serial link) 1,544 kbits/s, (serial link) 1,544 kbits/s, (Ethernet segment) 10,000 kbits/s
    Smallest bandwidth in the path: 1,544
    _BandW_ = 10,000,000/1,544 = 6,476
    _Delay_ = 2,000 + 2,000 + 100 = 4,100
    _Metric_ = _BandW_ + _Delay_ = 10,576

Using the lower metric for the path via _Chicago_, _NewYork_’s route to `172.16.100.0` shows as:

```
NewYork>sh ip route
...
I       172.16.50.0 [100/1] via 172.16.250.2, 0:00:31, Serial0
I       172.16.100.0 [100/**`10576`**] via 172.16.250.2, 0:00:31, Serial0
I       172.16.252.0 [100/1] via 172.16.250.2, 0:00:31, Serial0
```

Let’s corroborate IGRP’s selection of the two-hop T-1 path in preference to the one-hop 56-kbps link by comparing the transmission delay for a 1,000-octet packet. A 1,000-octet packet will take 143 ms (1,000 x 8/56,000 second) over a 56-kbps link and 5 ms (1,000 x 8/1,544,000 second) over a T-1 link. Neglecting buffering and processing delays, two T-1 hops will cost 10 ms in comparison to 143 ms via the 56-kbps link.

#### 2.2.2.7 Processing IGRP updates

The processing of IGRP updates is very similar to the processing of RIP updates, described in [Chapter 2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02.html "Chapter 2. Routing Information Protocol (RIP)"). The IGRP update comes with an autonomous system number. If this does not match the IGRP AS number configured on the router receiving the update, the entire upgrade is disregarded. Thus, routers _NewYork_ and _Nairobi_ in TraderMary’s network will receive updates from each other but will discard them.

Each network number received in the update is checked for validity. Illegal network numbers such as `0.0.0.0/8`, `127.0.0.0/8`, and `128.0.0.0/16` are sometimes referred to as “Martian Network Numbers” and will be disregarded when received in an update (RFCs 1009, 1122).

The rules for processing IGRP updates are:

1. If the destination network number is unknown to the router, install the route using the source IP address of the update (provided the route is not indicated as unreachable).
    
2. If the destination network number is known to the router but the update contains a smaller metric, modify the routing table entry with the new next hop and metric.
    
3. If the destination network number is known to the router but the update contains a larger metric, ignore the update.
    
4. If the destination network number is known to the router and the update contains a higher metric that is from the same next hop as in the table, update the metric.
    
5. If the destination network number is known to the router and the update contains the same metric from a different next hop, install the route as long as the maximum number of paths to the same destination is not exceeded. These parallel paths are then used for load balancing. Note that the default maximum number of paths to a single destination is six in IOS Releases 11.0 or later.
    

### 2.2.3 Parallel Paths

For the routing table to be able to install multiple paths to the same destination, the IGRP metric for all the paths must be equal. The routing table will install several parallel paths to the same destination (the default maximum is six in current releases of IOS).

Load-sharing over parallel paths depends on the switching mode. If the router is configured for _process switching_ , load balancing will be on a packet-by-packet basis. If the router is configured for _fast switching_, load balancing will be on a per-destination basis. For a more detailed discussion of switching mode and load balancing, see [Chapter 2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02.html "Chapter 2. Routing Information Protocol (RIP)").

#### Unequal metric (cost) load balancing

The default behavior of IGRP installs parallel routes to a destination only if all routes have identical metric values. Traffic to the destination is load-balanced over all installed routes, as described earlier.

Equal-cost load balancing works well almost all the time. However, consider TraderMary’s network again. Say that TraderMary adds a node in London. Since traffic to London is critical, the network is engineered with two links from New York: one running at 128 kbps and another running at 56 kbps. [Figure 3-4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-FIG-4 "Figure 3-4. Unequal-cost load balancing") shows unequal-cost load balancing.

![[Pasted image 20241110172138.png]]
Figure 3-4. Unequal-cost load balancing

The routers are first configured as follows:

```
hostname NewYork
...
interface Ethernet0
ip address 172.16.1.1 255.255.255.0
!
interface Ethernet1
ip address 192.168.1.1 255.255.255.0
...
interface Serial2
bandwidth 128
ip address 172.16.249.1 255.255.255.0
!
interface Serial3
bandwidth 56
ip address 172.16.248.1 255.255.255.0
...
router igrp 10
network 172.16.0.0


hostname London
...
interface Ethernet0
ip address 172.16.180.1 255.255.255.0
!
interface Serial0
bandwidth 128
ip address 172.16.249.2 255.255.255.0
!
interface Serial1
bandwidth 56
ip address 172.16.284.2 255.255.255.0
...
router igrp 10
network 172.16.0.0
```

However, if you check _NewYork_’s routing table you will see that all traffic to London is being routed via the 128-kbps link:

```
NewYork>sh ip route
...
172.16.0.0/24 is subnetted, ...
I       172.16.180.0 [100/80225] via 172.16.249.2, 00:01:07, Serial2
...
```

This is because the _NewYork_ → _London_ metric is 80,225 via the 128-kbps path and 180,671 via the 56-kbps path.

The problem with this routing scenario is that the 56-kbps link is entirely unused, even when the 128-kbps link is congested. Overseas links are expensive: the network design ought to try to utilize all links. One way around this problem is to modify the IGRP parameters to make both links look equally attractive. This can be accomplished by modifying the 56-kbps path as follows:

```
hostname NewYork
...
interface Serial3
bandwidth 128
ip address 172.16.248.1 255.255.255.0
...
```

With this approach, both links would appear equally attractive. The routing table for _NewYork_ will look like this:

```
NewYork>sh ip route
...
172.16.0.0/24 is subnetted, ...
I       172.16.180.0 [100/80225] via 172.16.249.2, 00:01:00, Serial2
                     [100/80225] via 172.16.248.2, 00:01:00, Serial3
```

However, traffic will now be evenly distributed over the two links, which may congest the 56-kbps link while leaving the 128-kbps link underutilized.

Another solution is to modify IGRP’s default behavior and have it install unequal-cost links in its table, balancing traffic over the links in proportion to the metrics on the links. The variance that is permitted between the lowest and highest metrics is specified by an integer in the **variance** command. For example:

```
router igrp 10
network 172.16.0.0
variance 2
```

specifies that IGRP will install routes with different metrics as long as the largest metric is less than twice the lowest metric. In other words, if the variance is v, then:

highest metric ≥ lowest metric x v

The maximum number of routes that IGRP will install will still be four, by default. This maximum can be raised to six when running IOS 11.0 or later.

Going back to TraderMary’s network, the metric value for the 128-kbps path to London is 80,225 while the metric value for the 56-kbps path is 180,671. The ratio 180,671/80,225 is 2.25; hence, a variance of 3 will be adequate. _NewYork_ may now be configured as follows:

```
hostname NewYork
...
interface Ethernet0
ip address 172.16.1.1 255.255.255.0
!
interface Ethernet1
ip address 192.168.1.1 255.255.255.0
...
interface Serial2
bandwidth 128
ip address 172.16.249.1 255.255.255.0
!
interface Serial3
bandwidth 56
ip address 172.16.248.1 255.255.255.0
...
router igrp 10
network 172.16.0.0
variance 3
```

And the routing table for _NewYork_ will look like this:

```
NewYork>sh ip route
...
172.16.0.0/24 is subnetted, ...
I       172.16.180.0 [100/80225] via 172.16.249.2, 00:01:00, Serial2
                     [100/180671] via 172.16.248.2, 00:01:00, Serial3
```

Traffic from _NewYork_ to _London_ will be divided between _Serial2_ and _Serial3_ in the inverse ratio of their metrics: _Serial2_ will receive 2.25 times as much traffic as _Serial3_.

The default value of variance is 1. A danger with using a variance value of greater than 1 is the possibility of introducing a routing loop. Thus, _NewYork_ may start routing to _London_ via _Chicago_ if the variance is made sufficiently large. IGRP checks that the paths it chooses to install are always downstream (toward the destination) by choosing only next hops with lower metrics to the destination.

### 2.2.4 Steady State

It is important for you as the network administrator to be familiar with the state of the network during normal conditions. Deviations from this state will be your clue to troubleshooting the network during times of network outage. This output shows the values of the IGRP timers:

```
   NewYork#sh ip protocol
   Routing Protocol is "igrp 10"
     **`Sending updates every 90 seconds, next due in 61 seconds`**
     **`Invalid after 270 seconds, hold down 280, flushed after 630`**
     Outgoing update filter list for all interfaces is
     Incoming update filter list for all interfaces is
     Default networks flagged in outgoing updates
     Default networks accepted from incoming updates
     IGRP metric weight K1=1, K2=0, K3=1, K4=0, K5=0
     IGRP maximum hopcount 100
     IGRP maximum metric variance 1
     Redistributing: igrp 10
     Routing for Networks:
       172.16.0.0
     Routing Information Sources:
       Gateway         Distance      Last Update
1      172.16.250.2         100      00:00:40
2      172.16.251.2         100      00:00:09
     Distance: (default is 100)
```

Note that IGRP updates are sent every 90 seconds and the next update is due in 61 seconds, which means that an update was issued about 29 seconds ago.

Further, lines 1 and 2 show the gateways from which router _NewYork_ has been receiving updates. This list is valuable in troubleshooting -- missing routes from a routing table could be because the last update from a gateway was too long ago. Check the time of the last update to ensure that it is within the IGRP update timer:

```
NewYork#show ip route
...
Gateway of last resort is not set

     172.16.0.0/24 is subnetted, 6 subnets
I       172.16.252.0 [100/10476] via 172.16.251.2, 00:00:26, Serial1
                     [100/10476] via 172.16.250.2, 00:00:37, Serial0
C       172.16.250.0 is directly connected, Serial0
C       172.16.251.0 is directly connected, Serial1
I       172.16.50.0 [100/8576] via 172.16.250.2, 00:00:37, Serial0
C       172.16.1.0 is directly connected, Ethernet0
I       172.16.100.0 [100/8576] via 172.16.251.2, 00:00:26, Serial1
C    192.168.1.0/24 is directly connected, Ethernet1
```

One key area to look at in the routing table is the timer values. The format that Cisco uses for timers is _hh:mm:ss_ (hours:minutes:seconds). You would expect the time against each route to be between 00:00:00 (0 seconds) and 00:01:30 (90 seconds). If a route was received more than 90 seconds ago, that indicates a problem in the network. You should begin by checking to see if the next hop for the route is reachable.

You should also be familiar with the number of major network numbers (two in the previous output -- 172.16.0.0 and 192.168.1.0) and the number of subnets in each (six in 172.16.0.0 and one in 192.168.1.0). In most small to mid-sized networks, these counts will change only when networks are added or subtracted.

  

---
 The concept of an _outgoing_ interface is best illustrated with an example. In TraderMary’s network, the outgoing interfaces from _NewYork_ to 172.16.100.0 will be _NewYork_ interface _Serial0_, _Chicago_ interface _Serial_, and _Ames_ interface _Ethernet0_. When computing the metric for _NewYork_ to 172.16.100.0, we will use the IGRP parameters of bandwidth, delay, load, reliability, and MTU for these interfaces. We will not use the IGRP parameters from interfaces. However, unless they have been modified, the parameters on this second set of interfaces would be identical to the first.
 
## 2.3 Speeding Up Convergence

Like RIP, IGRP implements hold-downs, split horizon, triggered updates, and poison reverse (see [Chapter 2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02.html "Chapter 2. Routing Information Protocol (RIP)") for details on these convergence methods). Like RIP, IGRP also maintains an update timer, an invalid timer, a hold-down timer, and a flush timer for every route in the routing table:

Update timer (default value: 90 seconds)
	After sending a routing update, IGRP sets the update timer to 0. When the timer expires, IGRP issues another routing update.

Invalid timer (default value: 270 seconds)
	Every time a router receives an update for a route, it sets the invalid timer to 0. The expiration of the invalid timer indicates that the source of the routing information is suspect. Even though the route is declared invalid, packets are still forwarded to the next hop specified in the routing table. Note that prior to the expiration of the invalid timer, IGRP would process any updates received by updating the route’s timers.

Hold-down timer (default value: 280 seconds)
	When the invalid timer expires, the route automatically enters the hold-down phase. During hold-down all updates regarding the route are disregarded -- it is assumed that the network may not have converged and that there may be bad routing information circulating in the network. The hold-down timer is started when the invalid timer expires.

Flush timer (default value: 630 seconds)
	Every time a router receives an update for a route, it sets the flush timer to 0. When the flush timer expires, the route is removed from the routing table and the router is ready to receive a new route update. Note that the flush timer overrides the hold-down timer.

### 2.3.1 Setting Timers

IGRP timers can be modified to allow faster convergence. The configuration:

```
router igrp 10
timers basic 30 90 90 180
```

would generate IGRP updates every 30 seconds, mark a route invalid in 90 seconds, keep the route in hold-down for 90 seconds, and flush the route in 180 seconds.

However, IGRP timers should not be modified without a detailed understanding of route convergence in Distance Vector protocols (see [Chapter 2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02.html "Chapter 2. Routing Information Protocol (RIP)")). Selecting too short a hold-down period, for example, may cause bad routing information to persist in a network. Selecting too long a hold-down period would increase the time it takes to learn a route via a different path after a failure.

Changing timers also presents the danger that sooner or later someone will configure a router with default timers. This may cause _route flapping_; i.e., routes to some network numbers may become intermittently invisible.

#### Warning

Do not modify IGRP timers unless absolutely necessary. If you modify IGRP timers, make sure that all routers have the same timers.

### 2.3.2 Disabling IGRP Hold-Downs

IGRP hold-downs can be disabled with the command:

```
router igrp 10
no metric holddown
```

thus speeding up convergence when a route fails. However, the problem with turning off hold-downs is that if a triggered update regarding the failure does not reach some router, that router could insert bad routing information into the network. Doesn’t this seem like a dangerous thing to do?

Split horizon, triggered updates, and poison reverse are implemented in IGRP much like they are in RIP.

## 2.4 Route Summarization

IGRP summarizes network numbers when crossing a major network-number boundary, just like RIP does. Route summarization reduces the number of routes that need to be exchanged, processed, and stored.

However, route summarization does not work well in discontiguous networks. Consider the discontiguous network in [Figure 3-5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s04.html#iprouting-CHP-3-FIG-5 "Figure 3-5. Contiguous and discontiguous networks"). Router _X_ will receive advertisements for `10.0.0.0` from both routers _A_ and _B_. If _X_ sent packets with the destination `10.1.1.1` to _B_, the packet would be lost -- _B_ would have to drop the packet because it would not have a route for `10.1.1.1` in its table. Likewise, if _X_ sent packets with the destination `10.2.1.1` to _A_, the packet would be lost -- _A_ would have to drop the packet because it would not have a route for `10.2.1.1`.

![[Pasted image 20241110172649.png]]
Figure 3-5. Contiguous and discontiguous networks

Both IGRP and RIP networks must be designed in contiguous blocks of major network numbers.

## 2.5 Default Routes

IGRP tracks default routes in the exterior section of its routing updates. A router receiving `10.0.0.0` in the exterior section of a routing update would mark `10.0.0.0` as a default route and install its next hop to `10.0.0.0` as the _gateway of last resort_ . Consider the network in [Figure 3-6](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s05.html#iprouting-CHP-3-FIG-6 "Figure 3-6. Branch offices only need a default route") as an example in which a core router connects to several branch routers in remote sites.

![[Pasted image 20241110172821.png]]
Figure 3-6. Branch offices only need a default route

The core router is configured as follows:

```
   hostname core1
   !
   interface Ethernet0
    ip address 192.168.1.1 255.255.255.0
   ...
   interface Serial0
   ip address 172.16.245.1 255.255.255.0
   ...
   router igrp 10
3   redistribute static                
    network 172.16.0.0
4   default-metric 10000 100 255 1 1500             
   !
   no ip classless
5  ip default-network 10.0.0.0
6  ip route 10.0.0.0 255.0.0.0 Null0
```

The branch router is configured as follows:

```
hostname branch1
...
interface Serial0
ip address 172.16.245.2 255.255.255.0
...
router igrp 10
redistribute static
network 172.16.0.0
!
no ip classless
```

An examination of _branch1_’s routing table would show:

```
branch1#sh ip route
Codes: C - connected, S - static, I - IGRP, R - RIP, M - mobile, B - BGP
       D - EIGRP, EX - EIGRP external, O - OSPF, IA - OSPF inter area
       N1 - OSPF NSSA external type 1, N2 - OSPF NSSA external type 2
       E1 - OSPF external type 1, E2 - OSPF external type 2, E - EGP
       i - IS-IS, L1 - IS-IS level-1, L2 - IS-IS level-2, * - candidate default

Gateway of last resort is 172.16.245.1 to network 10.0.0.0

     172.16.0.0/24 is subnetted, 1 subnets
C       172.16.245.0 is directly connected, Serial0
I*   10.0.0.0/8 [100/8576] via 172.16.245.1, 00:00:26, Serial0
```

Note that network 10.0.0.0 has been flagged as a default route (*). To ensure that the default route works, let’s do a test to see if _branch1_ can ping 192.168.1.1, even though 192.168.1.0 is not in _branch1_’s routing table:

```
branch1#ping 192.168.1.1

Type escape sequence to abort.
Sending 5, 100-byte ICMP Echos to 192.168.1.1, timeout is 2 seconds:
!!!!!
Success rate is 100 percent (5/5), round-trip min/avg/max = 40/50/80 ms
```

Here are the steps we followed in the creation of the default route:

1. Network `10.0.0.0` was flagged as a default route by _core1_ (line 5).
    
2. Network `10.0.0.0` was defined via a static route (line 6).
    
3. The default route was redistributed into IGRP, which then placed the route in the exterior section of its update message to _branch1_ (line 3).
    
4. A default metric was attached to the redistribution (line 4).
    

There are a few things to note when creating default routes in IGRP. First, IGRP does not use `0.0.0.0` as a default route. Thus, if `0.0.0.0` were defined in place of `10.0.0.0`, IGRP would not convey it. Second, how should one choose which network number to flag as a default route? In the previous example, the network `10.0.0.0` does not need to be a real network number configured on an interface; it could just be a fictitious number (that does not exist as a real number in the network) to which all default traffic will be sent. Using a fictitious number instead of a real network number as the default route can have certain advantages. For example, a fictitious network number will not go down if an interface goes down. Further, changing the ideal candidate for the default route can be much easier with fictitious network numbers than with real network numbers.

### 2.5.1 Multiple Default Routes

To increase the reliability of the connection to branches, each branch may be connected to two core routers:

```
hostname core2
!
interface Ethernet0
 ip address 192.168.1.1 255.255.255.0
...
interface Serial0
ip address 172.16.246.1 255.255.255.0
...
router igrp 10
 redistribute static
 network 172.16.0.0
 default-metric 10000 100 255 1 1500
!
no ip classless
ip default-network 10.0.0.0
ip route 10.0.0.0 255.0.0.0 Null0
```

_branch1_ will now receive two default routes:

```
branch1>sh ip route
...
Gateway of last resort is 172.16.250.1 to network 10.0.0.0

     172.16.0.0/24 is subnetted, 2 subnets
C       172.16.245.0 is directly connected, Serial1
C       172.16.246.0 is directly connected, Serial0
I*   10.0.0.0/8 [100/8576] via 172.16.245.1, 00:00:55, Serial0
                [100/8576] via 172.16.246.1, 00:00:55, Serial1
```

Note that it is also possible to set up one router (say, _core1_) as primary and the second router as backup. To do this, set up the default from _core2_ with a _worse_ metric, as shown in line 7:

```
   hostname core2
   !
   interface Ethernet0
    ip address 192.168.1.1 255.255.255.0
   ...
   interface Serial0
   ip address 172.16.246.1 255.255.255.0
   ...
   router igrp 10
    redistribute static
    network 172.16.0.0
7   default-metric 1544 2000 255 1 1500           
   !
   no ip classless
   ip default-network 10.0.0.0
   ip route 10.0.0.0 255.0.0.0 Null0
```

## 2.6 Classful Route Lookups

Router _branch1_ is configured to perform classful route lookups (see line 7 in the previous code block). A classful route lookup works as follows:

1. Upon receiving a packet, the router first determines the major network number for the destination. If the destination IP address is `172.16.1.1`, the major network number is `172.16.0.0`. If the destination IP address is `192.168.1.1`, the major network number is `192.168.1.0`.
    
2. Next, the router checks to see if this major network number exists in the routing table. If the major network number exists in the routing table (`172.16.0.0` does), the router checks for the destination’s subnet. In our example, _branch1_ would look for the subnet `172.16.1.0`. If this subnet exists in the table, the packet will be forwarded to the next hop specified in the table. If the subnet does not exist in the table, the packet will be dropped.
    
3. If the major network number does not exist in the routing table, the router looks for a default route. If a default route exists, the packet will be forwarded as specified by the default route. If there is no default route in the routing table, the packet will be dropped.
    

Router _branch1_ is able to ping `192.168.1.1` as a consequence of rule 3:

```
branch1#ping 192.168.1.1

Type escape sequence to abort.
Sending 5, 100-byte ICMP Echos to 192.168.1.1, timeout is 2 seconds:
!!!!!
Success rate is 100 percent (5/5), round-trip min/avg/max = 40/50/80 ms
```

However, let’s define a new subnet of `172.16.0.0` on _core1_ (and then block the advertisement of this subnet with an access list on lines 8 and 9) and see if _branch1_ can reach it using a default route:

```
   hostname core1
   !
   interface Ethernet0
    ip address 192.168.1.1 255.255.255.0
   !
   interface Ethernet1
   ip address 172.16.10.1 255.255.255.0
   ...
   interface Serial0
   ip address 172.16.245.1 255.255.255.0
   ...
   router igrp 10
    redistribute static
    network 172.16.0.0
    default-metric 10000 100 255 1 1500
    distribute-list 1 out serial0
   !
   no ip classless
   ip default-network 10.0.0.0
   ip route 10.0.0.0 255.0.0.0 Null0
   !
8  access-list 1 deny 172.16.10.0 0.0.0.255   
9  access-list 1 permit 0.0.0.0 255.255.255.255


   branch1#sh ip route
   ...
   Gateway of last resort is 172.16.245.1 to network 10.0.0.0

        172.16.0.0/24 is subnetted, 1 subnets
   C       172.16.245.0 is directly connected, Serial0
   I*   10.0.0.0/8 [100/8576] via 172.16.245.1, 00:00:26, Serial0


   branch1#ping 192.168.1.1

   Type escape sequence to abort.
   Sending 5, 100-byte ICMP Echos to 172.16.10.1, timeout is 2 seconds:
   !!!!!
   Success rate is 100 percent (5/5), round-trip min/avg/max = 40/50/80 ms
```

This demonstrates the use of rule 2, which causes the packet for `172.16.10.1` to be dropped. Note that in this example `172.16.10.1` did not match the default route, whereas `192.168.1.1` did match the default.

Classless route lookup, the other option, is discussed in [Chapter 5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch05.html "Chapter 5. Routing Information Protocol Version 2 (RIP-2)").

## 2.7 Summing Up

IGRP has the robustness of RIP but adds a major new feature -- route metrics based on bandwidth and delay. This feature -- along with the ease with which it can be configured and deployed -- has made IGRP tremendously popular for small to mid-sized networks. However, IGRP does not address several problems that also affect RIP:

- The exchange of full routing updates does not scale for large networks -- the overhead of generating and processing all routes in the AS can be high.
    
- IGRP convergence times can be too long.
    
- Subnet mask information is not exchanged in IGRP updates, so Variable Length Subnet Masks (VLSM) and discontiguous address spaces are not supported.
    

These issues may be too significant to overlook in large IP networks in which address-space conservation may necessitate VLSM, full route updates would be so large that they would consume significant network resources (serial links to branches tend to saturate quickly, and smaller routers may consume a lot of CPU power just to process all the routes at every update interval), and the convergence times may be too long because of the network diameter. Even small to mid-sized networks may choose not to implement IGRP if convergence time is an issue.

---
The definition of small, medium, and large IP networks can be discussed ad nauseam because of the number of variables involved (number of routers and routes, network bandwidth/utilization, network delay/latency, etc.), but rough measures are as follows: small -- a few dozen routers with up to a few hundred routes; medium -- a few hundred routers with a few thousand routes; large -- anything bigger than medium.

# Chapter 3. Enhanced Interior Gateway Routing Protocol (EIGRP)

The Enhanced Interior Gateway Routing Protocol (EIGRP), referred to as an advanced Distance Vector protocol, offers radical improvements over IGRP. Traditional DV protocols such as RIP and IGRP exchange periodic routing updates with all their neighbors, saving the best distance (or metric) and the vector (or next hop) for each destination. EIGRP differs in that it saves not only the best (least-cost) route but all routes, allowing convergence to be much quicker. Further, EIGRP updates are sent only upon a network topology change; updates are not periodic.

Getting EIGRP running is not much more difficult than getting IGRP running, as we will see in Section 4.1.

Even though EIGRP offers radical improvements over IGRP, there are similarities between the protocols. Like IGRP, EIGRP bases its metric on bandwidth, delay, reliability, load, and MTU (see Section 4.2).

The fast convergence feature in EIGRP is due to the Diffusing Update Algorithm (DUAL), discussed in Section 4.3.

EIGRP updates carry subnet mask information. This allows EIGRP to summarize routes on arbitrary bit boundaries, support classless route lookups, and allow the support of Variable Length Subnet Masks (VLSM). This is discussed in Section 4.4 and Section 4.5.

Setting up default routes in EIGRP is discussed in [Section 3.6]

Troubleshooting EIGRP can be tricky. This chapter ends with some troubleshooting tips in [Section 3.7]

EIGRP is a Cisco proprietary protocol; other router vendors do not support EIGRP. Keep this in mind if you are planning a multivendor router environment.

This chapter focuses on EIGRP’s enhancements over IGRP: the use of DUAL; and the use of subnet masks in updates, which in turn allow VLSM and route summarization at arbitrary bit boundaries. This chapter does not cover router metrics in detail or the concept of parallel paths. Those concepts have not changed much in EIGRP. I assume that the reader is familiar with IGRP.

## 3.1 Getting EIGRP Running

TraderMary’s network, shown in [Figure 4-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04.html#iprouting-CHP-4-FIG-1 "Figure 4-1. TraderMary’s network"), can be configured to run EIGRP as follows.

![[Pasted image 20241110174232.png]]
Figure 4-1. TraderMary’s network

Just like RIP and IGRP, EIGRP is a distributed protocol that needs to be configured on every router in the network:

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
description New York to Chicago link
ip address 172.16.250.1 255.255.255.0
!
interface Serial1
description New York to Ames link
bandwidth 56
ip address 172.16.251.1 255.255.255.0
...
**`router eigrp 10`**
**`network 172.16.0.0`**

hostname Chicago
...
interface Ethernet0
ip address 172.16.50.1 255.255.255.0
!
interface Serial0
description Chicago to New York link
ip address 172.16.250.2 255.255.255.0
!
interface Serial1
description Chicago to Ames link
ip address 172.16.252.1 255.255.255.0
...

**`router eigrp 10`**
**`network 172.16.0.0`**

hostname Ames
...
interface Ethernet0
ip address 172.16.100.1 255.255.255.0
!
interface Serial0
description Ames to Chicago link
ip address 172.16.252.2 255.255.255.0
!
interface Serial1
description Ames to New York link
bandwidth 56
ip address 172.16.251.2 255.255.255.0
...

**`router eigrp 10`**
**`network 172.16.0.0`**
```

The syntax of the EIGRP command is:

```
router eigrp _`autonomous-system-number`_
```

in global configuration mode. The networks that will be participating in the EIGRP process are then listed:

```
network 172.16.0.0
```

What does it mean to list the network numbers participating in EIGRP?

1. Router _NewYork_ will include directly connected `172.16.0.0` subnets in its updates to neighboring routers. For example, `172.16.1.0` will now be included in updates to the routers _Chicago_ and _Ames_.
    
2. _NewYork_ will receive and process EIGRP updates on its `172.16.0.0` interfaces from other routers running EIGRP 10. For example, _NewYork_ will receive EIGRP updates from _Chicago_ and _Ames_.
    
3. By exclusion, network `192.168.1.0`, connected to _NewYork,_ will not be advertised to _Chicago_ or _Ames_, and _NewYork_ will not process any EIGRP updates received on _Ethernet0_ (if there is another router on that segment).
    

The routing tables for _NewYork_, _Chicago_, and _Ames_ will show all `172.16.0.0` subnets. Here is _NewYork_’s table:

```
  NewYork#sh ip route
  Codes: C - connected, S - static, I - IGRP, R - RIP, M - mobile, B - BGP
         D - EIGRP, EX - EIGRP external, O - OSPF, IA - OSPF inter area 
         N1 - OSPF NSSA external type 1, N2 - OSPF NSSA external type 2
         E1 - OSPF external type 1, E2 - OSPF external type 2, E - EGP
         i - IS-IS, L1 - IS-IS level-1, L2 - IS-IS level-2, * - candidate default
       
  Gateway of last resort is not set

1      **`172.16.0.0/24 is subnetted, 6 subnets`** 
  D       172.16.252.0 [90/2681856] via 172.16.250.2, 00:18:54, Ethernet0/0
  C       172.16.250.0 is directly connected, Ethernet0/0
  C       172.16.251.0 is directly connected, Ethernet0/1
  D       172.16.50.0 [90/2195456] via 172.16.250.2, 00:18:54, Ethernet0/0
  C       172.16.1.0 is directly connected, Loopback0
  D       172.16.100.0 [90/2707456] via 172.16.250.2, 00:18:54, Ethernet0/0
  C    192.168.1.0/24 is directly connected, Loopback1
```

The EIGRP-derived routes in this table are labeled with a “D” in the left margin. Note that the routing table provides summary information (as in line 1). Line 1 contains subnet mask information (24 bits, or `255.255.255.0`) and the number of subnets in `172.16.0.0` (6).

In addition to the routing table, EIGRP builds another table called the _topology table_ :

```
   NewYork#sh ip eigrp topology
   IP-EIGRP Topology Table for process 10

   Codes: P - Passive, A - Active, U - Update, Q - Query, R - Reply,
          r - Reply status

   P 172.16.252.0/24, 1 successors, FD is 2681856
            via 172.16.250.2 (2681856/2169856), Serial0
            via 172.16.251.2 (46738176/2169856), Serial1
   P 172.16.250.0/24, 1 successors, FD is 2169856
            via Connected, Serial0
   P 172.16.251.0/24, 1 successors, FD is 46226176
            via Connected, Serial1
   P 172.16.50.0/24, 1 successors, FD is 2195456
            via 172.16.250.2 (2195456/281600), Serial0
   P 172.16.1.0/24, 1 successors, FD is 128256
            via Connected, Ethernet0
2   **`P 172.16.100.0/24, 1 successors, FD is 2707456`**
3            **`via 172.16.250.2 (2707456/2195456), Serial0`**
4            **`via 172.16.251.2 (46251776/281600), Serial1`**
```

This topology table shows two entries for _Ames_’s subnet, `172.16.100.0` (line 2). Only the lower-cost route (line 3) is installed in the routing table, but the second entry in the topology table (line 4) allows _NewYork_ to quickly converge on the less preferred path if the primary path fails.

Note that network `192.168.1.0`, defined on _NewYork_ interface _Ethernet1_, did not appear in the routing tables of _Chicago_ and _Ames_. To be propagated, `192.168.1.0` would have to be defined in a network statement under the EIGRP configuration on _NewYork_:

```
hostname NewYork
...
router eigrp 10
network 172.16.0.0
network 192.168.1.0
```

Each EIGRP process is identified by an autonomous system (AS) number, just like IGRP processes. Routers with the _same_ AS numbers will exchange routing information with each other, resulting in a _routing domain_ . Routers with dissimilar AS numbers will not exchange any routing information by default. However, routes from one routing domain can be leaked into another domain through the redistribution commands 

Compare the routing table in this section with the corresponding table for IGRP in [Chapter 2]. The essential contents are identical: the same routes with the same next hops. However, the route metrics look much bigger and the route update times are very high. IGRP routes would have timed out a while ago.

EIGRP metrics are essentially derived from IGRP metrics. The following section provides a quick summary.

## 3.2 EIGRP Metric

The EIGRP composite metric is computed exactly as the IGRP metric is and then multiplied by 256. Thus, the default expression for the EIGRP composite metric is:

_Metric =_ [_BandW +Delay_]x 256

where _BandW_ and _Delay_ are computed exactly as for IGRP (see [Section 2.2.2] in [Chapter 2]. In summary, _BandW_ is computed by taking the smallest bandwidth (expressed in kbits/s) from all outgoing interfaces to the destination (including the destination) and dividing 10,000,000 by this number (the smallest bandwidth), and _Delay_ is the sum of all the delay values to the destination network (expressed in tens of microseconds).

Further, note that the total delay (line 6), minimum bandwidth (line 6), reliability (line 7), minimum MTU (line 7), and load (line 8) for a path, which are used to compute the composite metric (line 5), are shown as output of the **show ip route destination-network-number** command:

```
   NewYork#sh ip route 172.16.50.0
   Routing entry for 172.16.50.0 255.255.255.0
     Known via "eigrp 10", distance 90, metric 2195456, type internal
     Redistributing via eigrp 10
     Last update from 172.16.250.2 on Serial0, 00:00:21 ago
     Routing Descriptor Blocks:
     * 172.16.50.0, from 172.16.250.2, 00:00:21 ago, via Serial0
5        **`Route metric is 2195456, traffic share count is 1`**
6        **`Total delay is 21000 microseconds, minimum bandwidth is 1544 Kbit`** 
7        **`Reliability 255/255, minimum MTU 1500 bytes`**
8        **`Loading 1/255, Hops 1`**
```

Converting route metrics between EIGRP and IGRP is very straightforward: EIGRP metrics are 256 times larger than IGRP metrics. This easy conversion becomes important when a network is running both IGRP and EIGRP, such as during a migration from IGRP to EIGRP.

Just like IGRP, EIGRP can be made to use load and reliability in its metric by modifying the parameters k1, k2, k3, k4, and k5 (see [Section 2.2.2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-SECT-2.2 "IGRP Metric") in [Chapter 2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03.html "Chapter 3. Interior Gateway Routing Protocol (IGRP)")).

The constants k1, k2, k3, k4, and k5 can be modified with the following command:

```
metric weights tos k1 k2 k3 k4 k5
```
### Warning

Cisco strongly recommends _not_ modifying the k1, k2, k3, k4, and k5 values for EIGRP.

## 3.3 How EIGRP Works

Unlike traditional DV protocols such as RIP and IGRP, EIGRP does not rely on _periodic_ updates: routing updates are sent only when there is a change. Remember that RIP and IGRP reset the invalid and flush timers upon receiving a route update. When a route is lost, the updates stop; the invalid and flush timers grow and grow (the timers are not reset), and, ultimately, the route is flushed from the routing table. This process of convergence assumes periodic updates. EIGRP’s approach has the advantage that network resources are not consumed by periodic updates. However, if a router dies, taking away all its downstream routes, how would EIGRP detect the loss of these routes? EIGRP relies on small _hello packets_ to establish neighbor relationships and to detect the loss of a neighbor. Neighbor relationships are discussed in detail in the next section.

RIP and IGRP suffer from a major flaw: _routing loops_ . Routing loops happen when information about the loss of a route does not reach all routers in the network because an update packet gets dropped or corrupted. These routers (that have not received the information about the loss of the route) inject bad routing information back into the network by telling their neighbors about the route they know. EIGRP uses _reliable_ transmission for all updates between neighbors. Neighbors acknowledge the receipt of updates, and if an acknowledgment is not received, EIGRP retransmits the update.

RIP and IGRP employ a battery of techniques to reduce the likelihood of routing loops: split horizon, hold-down timers, and poison reverse. These techniques do not guarantee that loops will not occur and, in any case, result in long convergence times. EIGRP uses the Diffusing Update Algorithm (DUAL) for all route computations. DUAL’s convergence times are an order of magnitude lower than those of traditional DV algorithms. DUAL is able to achieve such low convergence times by maintaining a table of loop-free paths to every destination, in addition to the least-cost path. DUAL is described in more detail later in this chapter.

DUAL can support IP, IPX, and AppleTalk. A protocol-dependent module encapsulates DUAL messages and handles interactions with the routing table. In summary, DUAL requires:

1. A method for the discovery of new neighbors and their loss (see the next section, [Section 3.3.1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.1 "Neighbor Relationship")).
    
2. Reliable transmission of update packets between neighbors (see the later section [Section 3.3.2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.2 "Reliable Transport Protocol")).
    
3. Protocol-dependent modules that can encapsulate DUAL traffic in IP, IPX, or AppleTalk. This text will deal only with EIGRP in IP networks (see the later section [Section 3.3.4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.4 "Protocol-Dependent Module")).
    

I’ll end this section with a discussion of EIGRP packet formats.

### 3.3.1 Neighbor Relationship

A router discovers a neighbor when it receives its first hello packet on a directly connected network. The router requests DUAL to send a full route update to the new neighbor. In response, the neighbor sends its full route update. Thus, a new neighbor relationship is established in the following steps:

1. When a router _A_ receives a hello packet from a new neighbor _B_, _A_ sends its topology table to router _B_ in unicast updates with the _initialization bit_ turned on.
    
2. When router _B_ receives a packet with the initialization bit on, it sends its topology table to router _A_.
    

The interval between hello packets from any EIGRP-speaking router on a network is five seconds (by default) on most media types. Each hello packet advertises _hold-time_ _--_ the length of time the neighbor should consider the sender up. The default hold-time is 15 seconds. If no hellos are received for the duration of the hold-time, DUAL is informed that the neighbor is down. Thus, in addition to detecting a new neighbor, hello packets are also used to detect the loss of a neighbor.

The hello-interval can be changed with the following command in interface configuration mode:

```
ip hello-interval eigrp _`autonomous-system-number seconds`_
```

Lengthening the hello-interval will also lengthen the route convergence time. However, a longer hello-interval may be desirable on a congested network with many EIGRP routers.

If the hello-interval is changed, the hold-time should also be modified. A rule of thumb is to keep the hold-time at three times the hello-interval.

```
ip hold-time eigrp _`autonomous-system-number seconds`_
```

Note that the hello-interval and hold-time need _not_ be the same for all routers on a network. Each router advertises its own hold-time, which is recorded in the neighbor’s neighbor table.

The default hello-interval is 60 seconds (with a hold-time of 180 seconds) on multipoint interfaces (such as ATM, Frame Relay, and X.25) with link speeds of T-1 or less. Hello packets are multicast; no acknowledgments are expected.

The following output shows _NewYork_’s neighbors. The first column -- labeled H -- is the order in which the neighbors were learned. The hold-time for `172.16.251.2` (_Ames_) is 10 seconds, from which we can deduce that the last hello was received 5 seconds ago. The hold-time for `172.16.250.2` (_Chicago_) is 13 seconds, from which we can deduce that the last hello was received 2 seconds ago. The hold-time for a neighbor should not exceed 15 seconds or fall below 10 seconds (if the hold-time fell below 10 s, that would indicate the loss of one or more hello packets).

```
NewYork#sh ip eigrp neighbor
IP-EIGRP neighbors for process 10
H   Address                 Interface   Hold Uptime   SRTT   RTO  Q  Seq
                                        (sec)         (ms)       Cnt Num
1   172.16.251.2            Se0/1         10 00:17:08   28  2604  0  7
0   172.16.250.2            Se0/0         13 00:24:43   12  2604  0  14.
```

After a neighbor relationship has been established between A and B the only EIGRP overhead is the exchange of hello packets, unless there is a topological change in the network.

### 3.3.2 Reliable Transport Protocol

The EIGRP transport mechanism uses a mix of multicast and unicast packets, using reliable delivery when necessary. All transmissions use IP with the protocol type field set to 88. The IP multicast address used is `224.0.0.10`.

DUAL requires guaranteed and sequenced delivery for some transmissions. This is achieved using acknowledgments and sequence numbers. So, for example, _update packets_ (containing routing table data) are delivered reliably (with sequence numbers) to all neighbors using multicast. _Acknowledgment packets_ _--_ with the correct sequence number -- are expected from every neighbor. If the correct acknowledgment number is not received from a neighbor, the update is retransmitted as a unicast.

The sequence number (seq num) in the last packet from the neighbor is recorded to ensure that packets are received in sequence. The number of packets in the queue that might need retransmission is shown as a queue count (QCnt), and the smoothed round trip time (SRTT) is used to estimate how long to wait before retransmitting to the neighbor. The retransmission timeout (RTO) is the time the router will wait for an acknowledgment before retransmitting the packet in the queue.

Some transmissions do not require reliable delivery. For example, hello packets are multicast to all neighbors on an Ethernet segment, whereas acknowledgments are unicast. Neither hellos nor acknowledgments are sent reliably.

EIGRP also uses _queries_ and _replies_ as part of DUAL. Queries are multicast or unicast using reliable delivery, whereas replies are always reliably unicast. Query and reply packets are discussed in more detail in the next section.

### 3.3.3 Diffusing Update Algorithm (DUAL)

All route computations in EIGRP are handled by DUAL. One of DUAL’s tasks is maintaining a table of loop-free paths to every destination. This table is referred to as the _topology table_ . Unlike traditional DV protocols that save only the best (least-cost) path for every destination, DUAL saves all paths in the topology table. The least-cost path(s) is copied from the topology table to the routing table. In the event of a failure, the topology table allows for very quick convergence if another loop-free path is available. If a loop-free path is not found in the topology table, a route recomputation must occur, during which DUAL queries its neighbors, who, in turn, may query their neighbors, and so on... hence the name “Diffusing” Update Algorithm.

These processes are described in detail in the following sections.

#### 3.3.3.1 Reported distance

Just like RIP and IGRP, EIGRP calculates the lowest cost to reach a destination based on updates from neighbors. An update from a router _R_ contains the cost to reach the destination network _N_ from _R_. This cost is referred to as the _reported distance_ (RD). _NewYork_ receives an update from _Ames_ with a cost of 281,600, which is _Ames_’s cost to reach `172.16.100.0`. In other words, the RD for _Ames_ to reach `172.160.100.0` as reported to _NewYork_ is 281,600. Just like _Ames_, _Chicago_ will report its cost to reach `172.16.100.0`. _Chicago_’s RD is 2,195,456 (see [Figure 4-2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-2 "Figure 4-2. Ames is a feasible successor for 172.16.100.0")).

![[Pasted image 20241110174908.png]]
Figure 4-2. Ames is a feasible successor for 172.16.100.0

#### 3.3.3.2 Feasible distance and successor

_NewYork_ will compute its cost to reach `172.16.100.0` via _Ames_ and _Chicago_. _NewYork_ will then compare the metrics for the two paths. _NewYork_’s cost via _Ames_ is 46,251,776. _NewYork_’s cost via _Chicago_ is 2,707,456. The lowest cost to reach a destination is referred to as the _feasible distance_ (FD) for that destination. _NewYork_’s FD to `172.16.100.0` is 2,707,456 (_BandW_ = 1,544 and _Delay_ = 4,100). The next-hop router in the lowest-cost path to the destination is referred to as the _successor_ . _NewYork_’s successor for `172.16.100.0` is `172.16.50.1` (_Chicago_).

#### 3.3.3.3 Feasibility condition and feasible successor

If a reported distance for a destination is less than the feasible distance for the same destination, the router that advertised the RD is said to satisfy the _feasibility condition_ (FC) and is referred to as a _feasible successor_ (FS). _NewYork_ sees an RD of 281,600 via _Ames_, which is lower than _NewYork_’s FD of 2,707,456. _Ames_ satisfies the FC. _Ames_ is an FS for _NewYork_ to reach `172.16.100.0`.

#### 3.3.3.4 Loop freedom

The feasibility condition is a test for _loop freedom_ : if the FC is met, the router advertising the RD must have a path to the destination not through the router checking the FC -- if it did, the RD would have been higher than the FD.

Let’s illustrate this concept with another example. Consider the network in [Figure 4-3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-3 "Figure 4-3. Loop freedom"). The metric values used in this example have been simplified to small numbers to make it easier to follow the concept.

![[Pasted image 20241110174948.png]]
Figure 4-3. Loop freedom

Router _A_’s best route to network _N_ is via router _B_, and the cost of this path is 100 (_A_’s FD to _N_ is 100). Router _X_ also knows how to get to network _N_; _X_ advertises _N_ to _A_ in an update packet (_A_ copies this information into its topology table). In the event that _A_’s link to _B_ fails, _A_ can use the route to _N_ via _X_ if _X_ does not use _A_ to get to _N_ (in other words, if the path is loop-free). Thus, the key question for _A_ to answer is whether or not the path that _X_ advertises is loop-free.

Here is how _A_ answers this question. Let’s say that _X_ advertises _N_ with a metric of 90 (_X_’s RD for _N_). _A_ compares 90 (RD) with 100 (FD). Is RD < FD? This comparison is the FC check. Since _A_’s FD is 100, _X_’s path to _N_ must not be via _A_ (and is loop-free). If _X_ advertises _N_ with a metric of 110, _X_’s path to _N_ could be via _A_ (the RD is not less than the FD, so the FC check fails) -- 110 could be _A_’s cost added to the metric of the link between _A_ and _X_ (and, hence, is not guaranteed to be free of a loop).

#### 3.3.3.5 Topology table

All destinations advertised by neighbors are copied into the topology table. Each destination is listed along with the neighbors that advertised the destination, the RD, and the metric to reach the destination via that neighbor. Let’s look at _NewYork_’s topology table and zoom in on destination `172.16.100.0`. There are two neighbors that sent updates with this destination: _Chicago_ (`172.16.250.2`) and _Ames_ (`172.16.251.2`), as shown on lines 9 and 10, respectively:

```
    NewYork#sh ip eigrp topology
    IP-EIGRP Topology Table for process 10

    Codes: P - Passive, A - Active, U - Update, Q - Query, R - Reply,
           r - Reply status

    ...
    P 172.16.100.0/24, 1 successors, FD is 2,707,456
9            **`via 172.16.250.2 (2,707,456/2,195,456), Serial0`**
10            **`via 172.16.251.2 (46,251,776/281,600), Serial1`**
```

_Chicago_ sent an update with an RD of 2,195,456, and _Ames_ sent an update with an RD _of_ 281,600. _NewYork_ computes its own metric to 172.16.100.0: 2,707,456 and 46,251,776 via _Chicago_ and _Ames_, respectively. _NewYork_ uses the lower-cost path via _Chicago_. _NewYork_’s FD to 172.16.100.0 is thus 2,707,456, and _Chicago_ is the successor. Next _NewYork_ checks to see if _Ames_ qualifies as a feasible successor. _Ames_’s RD is 281,600. This is checked against the FD. Since the RD < FD (281,600 < 2,707,456), _Ames_ is a feasible successor (see [Figure 4-2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-2 "Figure 4-2. Ames is a feasible successor for 172.16.100.0")).

Note that not all loop-free paths satisfy the FC. Thus, _NewYork_’s topology table does not contain the alternate path to 172.16.50.0 (via _Ames_). The FC guarantees that the paths that satisfy the condition are loop-free; however, not all loop-free paths satisfy the FC.

Let’s take a closer look at 172.16.50.0 (_Chicago_) in _NewYork_’s topology table:

```
NewYork#sh ip eigrp topology
IP-EIGRP Topology Table for process 10

Codes: P - Passive, A - Active, U - Update, Q - Query, R - Reply,
       r - Reply status

...
P 172.16.50.0/24, 1 successors, FD is 2195456
         via 172.16.250.2 (2195456/281600), Serial0
```

Notice that _Ames_ (`172.16.251.2`) did not become a feasible successor, even though _Ames_ offers a valid loop-free path. The condition that _Ames_ would have to satisfy to become a feasible successor is for its RD to be less than _NewYork_’s FD to `172.16.50.0`. _Ames_’s RD can be seen from _Ames’s_ routing table:

```
    Ames#sh ip route
    ...
         172.16.0.0/24 is subnetted, 6 subnets
    C       172.16.252.0 is directly connected, Serial0
    D       172.16.250.0 [90/2681856] via 172.16.252.1, 00:21:10, Serial0
    C       172.16.251.0 is directly connected, Serial1
11  D       172.16.50.0 [90/2195456] via 172.16.252.1, 00:21:10, Serial0
    D       172.16.1.0 [90/2707456] via 172.16.252.1, 00:15:36, Serial0
    C       172.16.100.0 is directly connected, Ethernet0
```

_Ames_’s metric to `172.16.50.0` is 2,195,456 (line 11). This will be the metric that _Ames_ reports to _NewYork_. The RD is thus 2,195,456. _NewYork_’s FD to `172.16.50.0` is 2,195,456. The RD and the FD are equal, which is not surprising given the topology: both _NewYork_ and _Ames_ have identical paths to `172.16.50.0` -- a T-1 link, a router, and the destination Ethernet segment. Since the condition for feasible successor is that RD < FD, _Ames_ is not an FS for `172.16.50.0` (see [Figure 4-4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-4 "Figure 4-4. Ames is not a feasible successor for 172.16.50.0")).

![[Pasted image 20241110175112.png]]
Figure 4-4. Ames is not a feasible successor for 172.16.50.0

The output of **show ip eigrp topology** shows only feasible successors. The output of **show ip eigrp topology all-links** shows all neighbors, whether feasible successors or not.

Note the “P” for “passive state” in the left margin of each route entry in _NewYork_’s topology table. _Passive state_ indicates that the route is in quiescent mode, implying that the route is known to be good and that no activities are taking place with respect to the route.

Any of the following events can cause DUAL to reevaluate its feasible successors:

- The transition in the state of a directly connected link
    
- A change in the metric of a directly connected link
    
- An update from a neighbor
    

If DUAL finds a feasible successor in its own topology table after one of these events, the route remains in passive state. If DUAL cannot find a feasible successor in its topology table, it will send a query to all its neighbors and the route will transition to _active state_ .

The next section contains two examples of DUAL reevaluating its topology table. In the first example, the route remains passive; in the second example, the route becomes active before returning to the passive state.

#### 3.3.3.6 Convergence in DUAL -- local computation

Let’s say that the _NewYork_ → _Chicago_ link fails ([Figure 4-5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-5 "Figure 4-5. Link failure")).

![[Pasted image 20241110175153.png]]
Figure 4-5. Link failure

_NewYork_’s routing table shows that `172.16.100.0` and `172.16.50.0` are learned via this link (_Serial0_):

```
NewYork#sh ip route
...
     172.16.0.0/24 is subnetted, 6 subnets
...
D       172.16.50.0 [90/2195456] via 172.16.250.2, 00:18:54, Serial0
D       172.16.100.0 [90/2707456] via 172.16.250.2, 00:18:54, Serial0
...
```

These routes become invalid. DUAL attempts to find new successors for both destinations -- `172.16.50.0` and `172.16.100.0`.

Let’s start with `172.16.100.0`. DUAL checks the topology table for `172.16.100.0`:

```
    NewYork#sh ip eigrp topology
    ...
    P 172.16.100.0/24, 1 successors, FD is 2707456
12            **`via 172.16.250.2 (2707456/2195456), Serial0`** 
13            **`via 172.16.251.2 (46251776/281600), Serial1`** 
```

Since _Serial0_ is down, the only feasible successor is `172.16.251.2` (_Ames_). Let’s review how _Ames_ qualifies as an FS. The FS check is:

- RD < FD.
    
- RD=281,600 (line 13).
    
- FD=2,707,456 (line 12).
    
- Since 281,600 < 2,707,456, _Ames_ qualifies as an FS.
    

In plain words, this implies that the path available to _NewYork_ via _Ames_ (the FS) is independent of the primary path that just failed. DUAL installs _Ames_ as the new successor for `172.16.100.0`.

In our case study, only one FS was available. In general, multiple FSs may be available, all of which satisfy the condition that their RD < FD, where FD is the cost of the route to the destination via the successor that was just lost.

DUAL will compute its metric to reach the destination via each FS. Since DUAL is searching for the successor(s) for this destination, it will choose the minimum from this set of metrics via each FS. Let the lowest metric be _Dmin_. If only one FS yields this metric of _Dmin_, that FS becomes the new successor. If multiple FSs yield metrics equal to _Dmin_, they all become successors (subject to the limitation in the maximum number of parallel paths allowed -- four or six, depending on the IOS version number). Since the new successor(s) is found locally (without querying any other router), the route stays in passive state. After DUAL has installed the new successor, it sends an update to all its neighbors regarding this change.

How long does this computation take? We simulated the failure of the _NewYork_ → _Chicago_ link in our laboratory. To measure how long EIGRP would take to converge after the failure of the link, we started a long ping test just before failing the _NewYork_ → _Chicago_ link:

```
NewYork#ping 
Protocol [ip]: 
Target IP address: 172.16.100.1
Repeat count [5]: 1000
...
Sending 1000, 100-byte ICMP Echos to 172.16.100.1, timeout is 2 seconds:

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Success rate is 99 percent (999/1000), round-trip min/avg/max = 1/3/92 ms
```

Note that only one ping packet was lost during this computation, implying that the convergence time (including the time to detect the failure of the link) was in the range of two to four seconds.

#### 3.3.3.7 Convergence in DUAL -- diffusing computation

Let’s next follow the steps that DUAL would take for `172.16.50.0`. Notice that this is a different case in that when _Serial0_ is down, _NewYork_ has no feasible successors in its topology table (see line 14).

```
   NewYork#sh ip eigrp topology
   ...
   P 172.16.50.0/24, 1 successors, FD is 2195456
14           **`via 172.16.250.2 (2195456/281600), Serial0`**    
   ...
```

DUAL knows of no feasible successors, but _NewYork_ has a neighbor that may know of a feasible successor. DUAL places the route in active state (see line 15) and sends a query to all its neighbors:

```
    NewYork#sh ip eigrp topology
    IP-EIGRP Topology Table for process 10

    Codes: P - Passive, A - Active, U - Update, Q - Query, R - Reply,
           r - Reply status

    ...
15   **`A 172.16.50.0/24, 0 successors, FD is 2195456, Q`**
        1 replies, active 00:00:06, query-origin: Local origin
        Remaining replies:
16            **`via 172.16.251.2, r, Serial1`**
```

which in this case is only `172.16.251.2` (_Ames_, as in line 16). _NewYork_ sets the reply flag on (line 16), which indicates that _NewYork_ expects a reply to the query. _Ames_ receives the query and marks its topology table entry for `172.16.50.0` via _NewYork_ as down. Next, _Ames_ checks its topology table for a feasible successor:

```
    Ames#sh ip eigrp topology
    IP-EIGRP Topology Table for process 10

    Codes: P - Passive, A - Active, U - Update, Q - Query, R - Reply,
           r - Reply status

    ...
17   **`P 172.16.50.0/24, 1 successors, FD is 2195456`**   
             via 172.16.252.1 (2195456/281600), Serial0
    ...
```

and finds that it has a successor (`172.16.252.1`). _Ames_ sends a reply packet to _NewYork_ with an RD of 2,195,456 (line 17). _NewYork_ marks the route as passive and installs a route for `172.16.50.0` via `172.16.251.2` (_Ames_).

In general, if DUAL does not find a feasible successor, it forwards the query to its neighbors. The query thus propagates (“diffuses”) until a reply is received. Routers that did not find a feasible successor would return an unreachable message. So, if _Ames_ did not have a feasible successor in its topology table, it would mark the route as active and propagate the query to its neighbor, if it had another neighbor. If _Ames_ had no other neighbor (and no feasible successor) it would return an unreachable message to _NewYork_ and mark the route as unreachable in its own table.

When DUAL marks a route as active and sets the _r_ flag on, it sets a timer for how long it will wait for a reply. The default value of the timer is three minutes. DUAL waits for a reply from all the neighbors it queries. If a neighbor does not respond to a query, the route is marked as _stuck-in-active_ and DUAL deletes all routes in its topology table that point to the unresponsive neighbor as a feasible successor.

### 3.3.4 Protocol-Dependent Module

The successors in the DUAL topology table are eligible for installation in the routing table. Successors represent the best path to the destination known to DUAL. However, whether the successor is copied into the routing table is another matter. The router may be aware of a route to the same destination from another source (such as another routing protocol or via a static route) with a lower _distance_. The IP protocol-dependent module (PDM) handles this task. The PDM may also carry information in the reverse direction -- from the routing table to the topology table. This will occur if routes are being redistributed into EIGRP from another protocol.

The PDM is also responsible for encapsulating EIGRP messages in IP packets.

### 3.3.5 EIGRP Packet Format

EIGRP packets are encapsulated directly in IP with the protocol field set to 88. The destination IP address in EIGRP depends on the packet type -- some packets are sent as multicast (with an address of `224.0.0.10`) and others are sent as unicast (see the earlier section [Section 3.3.2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.2 "Reliable Transport Protocol") for more details). The source IP address is the IP address of the interface from which the packet is issued.

Following the IP header is an EIGRP header. Key fields in the EIGRP header are as follows, and are also shown in [Figure 4-6](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-6 "Figure 4-6. Format of EIGRP packets"):

- The _opcode_ field specifies the EIGRP packet type (update, query, reply, hello).
    
- The _checksum_ applies to the entire EIGRP packet, excluding the IP header.
    
- The rightmost bit in the _flags_ field is the initialization bit and is used in establishing a new neighbor relationship (see [Section 3.3.1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.1 "Neighbor Relationship") earlier in this chapter).
    
- The _sequence_ and _ack_ fields are used to send messages reliably (see [Section 3.3.2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.2 "Reliable Transport Protocol") earlier in this chapter).
    
- The _AS number_ identifies the EIGRP process issuing the packet. The EIGRP process receiving the packet will process the packet only if the receiving EIGRP process has the same AS number; otherwise, the packet will be discarded.
    

![[Pasted image 20241110175413.png]]
Figure 4-6. Format of EIGRP packets

The fields following the EIGRP header depend on the opcode field. Of particular interest to routing engineers is the information in updates. We will ignore the other types of EIGRP messages and focus on IP internal route updates and IP external route updates.

_Internal_ routes contain destination network numbers learned within this EIGRP AS. For example, _NewYork_ learns `172.16.50.0` from EIGRP 10 on _Chicago_ as an internal route.

_External_ routes contain destination network numbers that were not learned within this EIGRP AS but rather derived from another routing process and redistributed into this EIGRP AS.

Internal and external routes are represented differently in the EIGRP update.

#### 3.3.5.1 Internal routes

Internal routes have a _type_ field of 0x0102. The metric information contained with the route is much like IGRP’s (see [Chapter 2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03.html "Chapter 3. Interior Gateway Routing Protocol (IGRP)")). However, there are two new fields: _next hop_ and _prefix length_. [Figure 4-7](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-7 "Figure 4-7. EIGRP internal route") shows the value field for the IP internal route.

![[Pasted image 20241110175451.png]]
Figure 4-7. EIGRP internal route

The next hop identifies the router to send packets destined for _destination_, the network number of the destination. In general, the next hop field for internal routes will be the IP address of the router on the interface on which it is issuing the update.

The prefix length field signifies the subnet mask to be associated with the network number specified in the destination field. Thus, if an EIGRP router is configured as follows:

```
ip address 172.16.1.1 255.255.255.0
```

it will advertise `172.16.1.0` with a prefix length of 24.

Likewise, if the router is configured as follows:

```
ip address 172.16.250.1 255.255.255.252
```

it will advertise `172.16.250.0` with a prefix length of 30.

#### 3.3.5.2 External routes

Additional fields are required to represent the source from which external routes are derived, as shown in [Figure 4-8](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-8 "Figure 4-8. EIGRP external route").

![[Pasted image 20241110175542.png]]
Figure 4-8. EIGRP external route

The next hop field identifies the router to send packets destined for _destination_, the network number of the destination. This field was absent in the IGRP update. Let’s look at what this field signifies.

In IGRP, if router _X_ sends an update to router _A_ with a destination network number of _N_, router _A_’s next hop for packets to _N_ will be _X_. In EIGRP, router _X_ can send an update to router _A_ with a destination network number of _N_ and a next hop field of _Y_. This is useful, say, in a scenario where _X_ and _Y_ are running RIP and _X_ is redistributing routes from RIP to IGRP. When _X_ sends an update to its neighbors on a shared network, _X_ can tell them to send traffic for network _N_ directly to _Y_ and not to _X_. This saves _X_ from having to accept traffic on a shared network and then reroute it to _Y_.

The _originating router, originating AS, external protocol metric_, and _external protocol ID_ fields specify information about the router and the routing process from which this route was derived. The external protocol ID specifies the routing protocol from which this route was derived. Here is a partial list of external protocol IDs: IGRP -- 0x01; EIGRP -- 0x02; RIP -- 0x04; OSPF -- 0x06; BGP -- 0x09. Thus, if a route was learned from RIP with a hop count of 3 and redistributed into EIGRP, the originating router field would contain the address of the RIP router, the originating AS field would be empty, the external protocol metric would be 3, and the external protocol ID would be 0x04.

The _arbitrary tag_ field is used to carry route maps.

Candidate default routes are marked by setting the flags field to 0x02. A flags field of 0x01 indicates an external route (but not a candidate default route).

The other parameters in the external route packet are similar to those in IGRP.

  

---

 Unlike RIP and IGRP, EIGRP updates are _not_ periodic. EIGRP updates are sent only when there is a topological change in the network.

You may ask why this cannot be handled by ICMP redirects. Cisco does not support redirects between routers.

## 3.4 Variable Length Subnet Masks

Unlike RIP and IGRP, EIGRP updates carry subnet mask information. The network architect now has the responsibility of using addresses wisely. Reviewing TraderMary’s configuration, a mask of `255.255.255.0` on the serial links is wasteful: there are only two devices on the link, so a 24-bit mask will waste 252 addresses. A 30-bit mask (`255.255.255.252`) allows two usable IP addresses in each subnet, which fits a serial line exactly.

Let’s say that the network architect decided to subdivide `172.16.250.0` using a 30-bit mask for use on up to 64 possible subnets. The subnets that thus become available are:

1. `172.16.250.0`
2. `172.16.250.4`
3. `172.16.250.8`
4. ...

1. `172.16.250.252`

The serial links in TraderMary’s network can be readdressed using these subnets:

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
description New York to Chicago link
**`ip address 172.16.250.1 255.255.255.252`**
!
interface Serial1
description New York to Ames link
bandwidth 56
**`ip address 172.16.250.5 255.255.255.252`**
...
router eigrp 10
network 172.16.0.0


hostname Chicago
...
interface Ethernet0
ip address 172.16.50.1 255.255.255.0
!
interface Serial0
description Chicago to New York link
**`ip address 172.16.250.2 255.255.255.252`**
!
interface Serial1
description Chicago to Ames link
**`ip address 172.16.250.9 255.255.255.0`**
...

router eigrp 10
network 172.16.0.0


hostname Ames
...
interface Ethernet0
ip address 172.16.100.1 255.255.255.0
!
interface Serial0
description Ames to Chicago link
ip address 172.16.250.10 255.255.255.0
!
interface Serial1
description Ames to New York link
bandwidth 56
ip address 172.16.250.6 255.255.255.0
...

router eigrp 10
network 172.16.0.0
```

_NewYork_’s routing table now looks like this:

```
NewYork#sh ip route
...
     172.16.0.0/16 is variably subnetted, 6 subnets, 2 masks
D       172.16.250.8/30 [90/2681856] via 172.16.250.2, 00:18:54, Serial0
C       172.16.250.0/30 is directly connected, Serial0
C       172.16.250.4/30 is directly connected, Serial1
D       172.16.50.0/24 [90/2195456] via 172.16.250.2, 00:18:54, Serial00
C       172.16.1.0/24 is directly connected, Ethernet0
D       172.16.100.0/24 [90/2707456] via 172.16.250.2, 00:18:54, Serial0
C    192.168.1.0/24 is directly connected, Ethernet1
```

Note that each route is now accompanied by its mask. When `172.16.0.0` had uniform masking, the routing table did not show the mask.

Further, let’s say that Casablanca is a small office with only a dozen people on the staff. We may safely assign Casablanca a mask of `255.255.255.192` (a limit of 62 usable addresses). Forward-thinking is important when assigning addresses. When running IGRP, the network architect may have had the foresight to assign addresses from the beginning of the range. Excess addresses should not be squandered, such as by randomly choosing addresses for hosts. A general rule is to start assigning addresses from the beginning or the bottom of an address range. When a site is shrinking, again keep all addresses at one end.

Using subnet masks that reflect the size of the host population conserves addresses. Put on your plate only as much as you will eat.

## 3.5 Route Summarization

The default behavior of EIGRP is to summarize on network-number boundaries. This is similar to RIP and IGRP and is a prudent way for a routing protocol to reduce the number of routes that are propagated between routers. However, there are some enhancements in the way EIGRP summarizes routes that merit a closer look.

### 3.5.1 Automatic Summarization

Say TraderMary’s network expands again, this time with a node in Shannon. Shannon gets connected to the London office via a 56-kbps link, as shown in [Figure 4-9](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s05.html#iprouting-CHP-4-FIG-9 "Figure 4-9. Route summarization").

![[Pasted image 20241110180531.png]]
Figure 4-9. Route summarization

Shannon has three Ethernet segments with an IP subnet on each: `172.20.100.0/24`, `172.20.101.0/24`, and `172.20.102.0/24`. The routers in London and Shannon are configured to run EIGRP 10 in keeping with the routing protocol in use in the U.S. _Shannon_ will advertise `172.20.0.0/16` to _London_ because the serial link from _London_ to _Shannon_ represents a network-number boundary (`172.20.0.0/172.16.0.0`). _Shannon_ itself will see all `172.16.0.0` subnets (without summarization) because it has a directly connected `172.16.0.0` network.

In EIGRP, the router doing the summarization will build a route to _null0_ (line 18) for the summarized address. Let’s check _Shannon_’s routing table:

```
    Shannon#sh ip route 172.20.0.0
    ...
         172.20.0.0/16 is subnetted, 6 subnets
    C       172.20.100.0/24 is directly connected, Ethernet0
    C       172.20.101.0/24 is directly connected, Ethernet1
18  **`D       172.20.0.0/16 is a summary, 00:12:11, Null0`**
    C       172.20.102.0/24 is directly connected, Ethernet2
```

The route to _null0_ ensures that if _Shannon_ receives a packet for which it has no route (e.g., `172.20.1.1`), it will route the packet using the null interface, thereby dropping the packet, rather than using some other route for the packet (such as a default route).

Now, let’s muddy the picture up a bit. TraderMary acquires a small company in Ottawa which also happens to use a `172.20.0.0` subnet -- `172.20.1.0`! The new picture looks something like [Figure 4-10](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s05.html#iprouting-CHP-4-FIG-10 "Figure 4-10. TraderMary’s networks in Shannon and Ottawa").

![[Pasted image 20241110180607.png]]
Figure 4-10. TraderMary’s networks in Shannon and Ottawa

Ottawa is also configured to run EIGRP 10 with a link from _NewYork_. Since the IP address on the link is `172.16.0.0`, _Ottawa_ will send a summary update of `172.20.0.0` to _NewYork_.

We have a problem now. There are two sources advertising `172.20.0.0`, and depending on where we are in the network, we will be able to route only to _Ottawa_ or _Shannon_. Thus, _NewYork_ will install `172.20.0.0` only via _Ottawa_, and _London_ will install `172.20.0.0` only via _Shannon_.

Unlike RIP and IGRP, EIGRP provides the option of disabling route summarization. Thus, _Shannon_ and _Ottawa_ can be configured as follows:

```
hostname Shannon
...
router eigrp 10
network 172.16.0.0
network 172.20.0.0
no auto-summary 


hostname Ottawa
...
router eigrp 10
network 172.16.0.0
network 172.20.0.0
no auto-summary
```

When _no auto-summary_ is turned on, _Shannon_ and _Ottawa_ will advertise their subnets to the rest of the network. The subnets happen to be unique, so any router will be able to route to any destination in the network.

Note that _no auto-summary_ was required only on the _Shannon_ and _Ottawa_ routers. _NewYork_ and _London_ and other routers will pass these subnets through (without summarizing them). Summarization happens only at a border between major network numbers, not at other routers.

The moral of this story is that EIGRP networks do not have to be contiguous with respect to major network numbers. However, I do not recommend deliberately building discontiguous networks. Summarizing on network-number boundaries is an easy way to reduce the size of routing tables and the complexity of the network. Disabling route summarization should be undertaken only when necessary.

### 3.5.2 Manual Summarization

EIGRP allows for the summarization of (external or internal) routes on any bit boundary. Manual summarization can be used to reduce the size of routing tables.

In our example, the network architect may decide to allocate blocks of addresses to _NewYork_, _Ames_, _Chicago_, etc. _NewYork_ is allocated the block `172.16.1.0` through `172.16.15.0`. This may also be represented as `172.16.0.0/20`, signifying that the first four bits of the third octet in this range are all zeros, as is true for `172.16.1.0` through `172.16.15.0`.

```
    hostname NewYork
    ...
19  **`interface Ethernet0`**
    **`ip address 172.16.1.1 255.255.255.0`**
    !
    interface Ethernet1
    ip address 192.168.1.1 255.255.255.0
    !
20  **`interface Ethernet2`**
    **`ip address 172.16.2.1 255.255.255.0`**
    !
    interface Serial0
    description New York to Chicago link
    ip address 172.16.250.1 255.255.255.0
    ip summary-address eigrp 10 172.16.0.0 255.255.240.0
    !
    interface Serial1
    description New York to Ames link
    bandwidth 56
    ip address 172.16.251.1 255.255.255.0
21  **`ip summary-address eigrp 10 172.16.0.0 255.255.240.0`**
    ...
    router eigrp 10
    network 172.16.0.0
```

_NewYork_ now has two Ethernet segments (lines 19 and 20) from this block and has also been configured to send a summary route for this block (line 21) to its neighbors. The configuration of these routers is as shown in [Figure 4-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04.html#iprouting-CHP-4-FIG-1 "Figure 4-1. TraderMary’s network"). Here’s _NewYork_’s routing table:

```
    NewYork#sh ip route
    ...
         172.16.0.0/16 is variably subnetted, 8 subnets, 2 masks
    D       172.16.252.0/24 [90/2681856] via 172.16.250.2, 00:01:44, Serial0
    C       172.16.250.0/24 is directly connected, Serial0
    C       172.16.251.0/24 is directly connected, Serial1
22  D       172.16.0.0/20 is a summary, 00:03:22, Null0 
    C       172.16.1.0/24 is directly connected, Ethernet0
    C       172.16.2.0/24 is directly connected, Ethernet2
    D       172.16.50.0/20 [90/2195456] via 172.16.250.2, 00:01:45, Serial0
    D       172.16.100.0/20 [90/2707456] via 172.16.250.2, 00:01:45, Serial0
    C    192.168.1.0/24 is directly connected, Ethernet1
```

Note that _NewYork_ installs a route to the null interface for the summarized address (`172.16.0.0/20`, as in line 22). Further, routers _Ames_ and _Chicago_ install this aggregated route (line 23) and not the individual `172.16.1.0/24` and `172.16.2.0/24` routes:

```
    Chicago#sh ip route
    ...
         172.16.0.0/16 is variably subnetted, 8 subnets, 2 masks
    C       172.16.252.0/24 is directly connected, Serial1
    C       172.16.250.0/24 is directly connected, Serial0
    D       172.16.251.0/24 [90/2681856] via 172.16.250.1, 00:02:30, Serial0
                            [90/2681856] via 172.16.252.2, 00:02:30, Serial1
    C       172.16.50.0/24 is directly connected, Ethernet0
23  D       172.16.0.0/20 [90/2195456] via 172.16.250.1, 00:02:12, Serial0   
    D       172.16.100.0/20 [90/2195456] via 172.16.252.2, 00:02:10, Serial1
```

The address aggregation commands on _NewYork_ reduce the routing-table size in the rest of the network. Note that address aggregation plans need to be laid out ahead of time so that network numbers can be allocated accordingly. Thus, in the previous example, _NewYork_ was allocated a block of 16 subnets:

`172.16.96.0` through `172.16.15.0`

Continuing this scheme, _Ames_ may be allocated a block of 16 addresses that envelop the network number it is currently using (`172.16.100.0`):

`172.16.96.0` through `172.16.111.0`

and _Chicago_ may be allocated a block of 16 addresses that envelop the network number it is currently using (`172.16.50.0`):

`172.16.48.0` through `172.16.63.0`

_Ames_ could now be configured to summarize its block using the statement on its serial interfaces:

```
ip summary-address eigrp 10 172.16.0.0 255.255.240.0
```

and _Chicago_ could be configured to summarize its block using the statement on its serial interfaces:

```
ip summary-address eigrp 10 172.16.0.0 255.255.240.0
```


---

If the subnets overlapped, disabling route summarization would not do us any good. There are other methods to tackle duplicate address problems, such as Network Address Translation (NAT).

## 3.6 Default Routes

EIGRP tracks default routes in the external section of its routing updates. Candidate default routes are marked by setting the flags field to 0x02.

Default routes are most often used to support branch offices that have only one or two connections to the core network (see [Figure 4-11](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s06.html#iprouting-CHP-4-FIG-11 "Figure 4-11. Branch offices only need a default route")).

![[Pasted image 20241110180948.png]]
Figure 4-11. Branch offices only need a default route

The core router is configured as follows:

```
    hostname core1
    !
    interface Ethernet0
     ip address 192.168.1.1 255.255.255.0
    ...
    interface Serial0
    ip address 172.16.245.1 255.255.255.0
    ...
    router eigrp 10
24   **`redistribute static metric 56 100 255 1 255`**   
     network 172.16.0.0
    !
    ip classless
25  **`ip route 0.0.0.0 0.0.0.0 Null0`** 
```

The branch router is configured as follows:

```
    hostname branch1
    ...
    interface Serial0
    ip address 172.16.245.2 255.255.255.0
    ...
26  **`router eigrp 10`**                                    
    network 172.16.0.0
```

An examination of _branch1_’s routing table would show:

```
    branch1#sh ip route
    ...
    Gateway of last resort is 172.16.251.1 to network 0.0.0.0

         172.16.0.0/24 is subnetted, 6 subnets
    C       172.16.245.0 is directly connected, Serial0
    ...
27  **`D*EX 0.0.0.0/0 [170/46251776] via 172.16.245.1, 00:01:47, Serial0`**
```

Since the default route is an external route, it is tagged with a distance of 170 (line 27).

The following steps were followed in the creation of this default route:

1. Network `0.0.0.0` was defined as a static route on _core1_ (see line 25).
2. Network `0.0.0.0` was redistributed into EIGRP 10 (see line 24).
3. A default metric was attached to the redistribution (line 24).
4. EIGRP 10 was turned on in _branch1_ (line 26).

To increase the reliability of the connection to branches, each branch may be connected to two core routers. _branch1_ will now receive two default routes. One router (say, _core1_) may be set up as the primary, and the second router (_core2_) as backup. To do this, set up the default from _core2_ with a _worse_ metric, as we did for IGRP in [Chapter 3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03.html "Chapter 3. Interior Gateway Routing Protocol (IGRP)").

## 3.7 Troubleshooting EIGRP

EIGRP can be difficult to troubleshoot because of its complexity. As a reminder, the best preparation for troubleshooting a network is to be familiar with the network and its state during normal (trouble-free) conditions. Become familiar with the routing tables, their sizes, the summarization points, routing timers, etc. Also, plan ahead with “what-if” scenarios. What if router _X_ failed or link _Y_ dropped? How would connectivity recover? Will all the routes still be in every router’s table? Will the routes still be summarized?

Perhaps the second-best preparation for troubleshooting a network is the ability to track network implementations and changes. If network implementations/changes are made in a haphazard way with no central control, an implementation team may walk away from a change (unaware that their change caused an outage) and it may take the troubleshooting team hours, or even days, to unravel the events that led to the outage. Besides making the network more vulnerable, such loose methods of network operation create bad relationships between teams.

The following sections are a partial list of network states/conditions to check when looking for clues to routing problems in EIGRP.

### Verifying Neighbor Relationships

If a router is unable to establish a stable relationship with its neighbors, it cannot exchange routes with those neighbors. The neighbor table can help check the integrity of neighbor relationships. Here is a sample of _NewYork_’s neighbor table:

```
NewYork#sh ip eigrp neighbor
IP-EIGRP neighbors for process 10
H   Address                 Interface   Hold Uptime   SRTT   RTO  Q  Seq
                                        (sec)         (ms)       Cnt Num
1   172.16.251.2            Se0/1         10 00:17:08   28  2604  0  7
0   172.16.250.2            Se0/0         13 00:24:43   12  2604  0  14
```

First, check that the neighbor count matches the number of EIGRP speakers. If routers _A_, _B_, and _C_ share an Ethernet segment and run EIGRP 10, all four routers should see each other in their neighbor tables. If router _C_ is consistently missing from _A_ and _B_’s tables, there may be a physical problem with _C_ or _C_ may be misconfigured (check _C_’s IP address and EIGRP configuration). Next, look for one-way neighbor relationships. Is _C_ in _A_ and _B_’s tables, but are _A_ and _B_ not in _C_’s table? This could indicate a physical problem with _C_’s connection or a filter that is blocking EIGRP packets.

If the hold-time exceeds 15 seconds (or the configured hold-time), the network may be congested and losing hellos. Increasing the hello-interval/hold-time may be a quick fix to the problem.

The uptime should reflect the duration that the routers have been up. A low uptime indicates that the neighbor relationship is being lost and reestablished.

The QCnt should be (or at least should not exceed on a consistent basis).

In summary, if a problem is found in the neighbor relationship, you should do the following:

1. Check for bad physical infrastructure.
2. Ensure that router ports are plugged into the correct hubs.
3. Check for filters blocking EIGRP packets.
4. Verify router configurations -- check IP addresses, masks, EIGRP AS numbers, and the network numbers defined under EIGRP.
5. Increase the hello-interval/hold-time on congested networks.

The command to clear and reestablish neighbor relationships is:

```
clear ip eigrp neighbors [_`ip address`_ | _`interface`_]
```
#### Tip

Repeatedly clearing all neighbor relationships causes the loss of routes (and the loss of packets to those routes). Besides, repeatedly issuing **clear** commands usually does not fix the problem.

### Stuck-in-Active

A route is regarded as stuck-in-active (SIA) when DUAL does not receive a response to a query from a neighbor for three minutes, which is the default value of the active timer. DUAL then deletes all routes from that neighbor, acting as if the neighbor had responded with an unreachable message for all routes.

Routers propagate queries through the network if feasible successors are not found, so it can be difficult to catch the culprit router (i.e., the router that is not responding to the query in time). The culprit may be running high on CPU utilization or may be connected via low-bandwidth links. Going back to TraderMary’s network, when _NewYork_ queries _Ames_ for `172.16.50.0`, it marks the route as active and lists the neighbor from which it is expecting a reply (line 28):

```
   NewYork#sh ip eigrp topology
   IP-EIGRP Topology Table for process 10

   Codes: P - Passive, A - Active, U - Update, Q - Query, R - Reply,
       r - Reply status

   ...
   A 172.16.50.0/24, 0 successors, FD is 2195456, Q
       1 replies, active 00:00:06, query-origin: Local origin
       Remaining replies:
28          via 172.16.251.2, r, Serial1
```

If this route were to become SIA, the network engineer should trace the path of the queries to see which router has been queried, has no outstanding queries itself, and yet is taking a long time to answer.

Starting from _NewYork_, the next router to check for SIA routes would be `172.16.251.2` (line 28). Finding the culprit router in large networks is a difficult task, because queries fan out to a large number of routers. Checking the router logs would give a clue as to which router(s) had the SIA condition.

#### Increase active timer

Another option is to increase the active timer. The default value of the active timer is three minutes. If you think the SIA condition is occurring because the network diameter is too large, with several slow-speed links (such as Frame Relay PVCs), it is possible that increasing the active timer will allow enough time for responses to return. The following command shows how to increase the active timer:

```
router eigrp 10
timers active-time minutes
```

For the change to be effective, the active timer must be modified on every router in the path of the query.

### EIGRP Bandwidth on Low-Speed Links

EIGRP limits itself to using no more than 50% of the configured bandwidth on router interfaces. There are two reasons for this:

1. Generating more traffic than the interface can handle would cause drops, thereby impairing EIGRP performance.
2. Generating a lot of EIGRP traffic would result in little bandwidth remaining for user data.

EIGRP uses the bandwidth that is configured on an interface to decide how much EIGRP traffic to generate. If the bandwidth configured on an interface does not match the physical bandwidth (the network architect may have put in an artificially low or high bandwidth value to influence routing decisions), EIGRP may be generating too little or too much traffic. In either case, EIGRP can encounter problems as a result of this. If it is difficult to change the **bandwidth** command on an interface because of such constraints, allocate a higher or lower percentage to EIGRP with the following command in interface configuration mode:

```
ip bandwidth percent eigrp _`AS-number percentage`_
```
### Network Logs

Check the output of the **show logging** command for EIGRP/DUAL messages. For example, the following message:

```
%DUAL-3-SIA: Route XXX stuck-in-active state in IP-EIGRP
```

indicates that the route _XXX_ was SIA.

### IOS Version Check, Bug Lists

The EIGRP implementation was enhanced in IOS Releases 10.3(11), 11.0(8), and 11.1(3) with respect to its performance on Frame Relay and other low-speed networks. In the event of chronic network problems, check the IOS versions in use in your network. Also use the bug navigation tools available on the Cisco web site.

### Debug Commands

As always, use **debug** commands in a production network only after careful thought. Having to resort to rebooting the router can be very unappetizing. The following is a list of EIGRP **debug** commands:

- **debug eigrp neighbors** (for neighbor-relationship activity)
- **debug eigrp packet** (all EIGRP packets)
- **debug eigrp ip neighbor** (if the previous two commands are used together, only EIGRP packets for the specified neighbor are shown)

## 3.8 Summing Up

EIGRP offers the following radical improvements over RIP and IGRP:

- Fast convergence -- convergence is almost instantaneous when a feasible successor is available.
- Variable Length Subnet Masks are supported -- subnet mask information is exchanged in EIGRP updates. This allows for efficient use of the address space, as well as support for discontiguous networks.
- Route summarization at arbitrary bit boundaries, reducing routing-table size.
- No regular routing updates -- network bandwidth and router CPU resources are not tied up in periodic routing updates, leading to improved network manageability.
- Ease of configuration -- EIGRP can be configured with almost the same ease as IGRP. However, troubleshooting DUAL can be difficult.

These EIGRP benefits come at the price of higher memory requirements (in addition to the routing table, EIGRP requires memory for the topology table and the neighbor table). DUAL is complex and can be very CPU-intensive, especially during periods of network instability when CPU resources are already scarce. Also, don’t forget that the EIGRP is a Cisco proprietary protocol.

EIGRP is in use today in several mid-sized networks.

# Chapter 4. Open Shortest Path First (OSPF)

Last year I flew from New York to Osaka for a conference. My journey began when I hailed a cab on Broadway in downtown New York. “JFK,” I told the cabbie, telling her my destination was John F. Kennedy Airport. I was still pushing my luggage down the seat so I could pull my door shut when the cab started to move. The cabbie changed lanes twice before I got it shut. I did make it to JFK in one piece, where I presented my ticket and boarded a flight to Osaka. At Osaka Airport, the taxi driver bowed to me as he took my luggage from my hand. Once the luggage was properly stowed, he asked for my destination. “New Otani Hotel,” I told him, and he bowed again and closed my side door.

This everyday story of a passenger in transit illustrates how a traveler is able to complete a journey in spite of the fact that the whereabouts of his destination are not known to every element in the system. The cabbie in New York knows only local destinations and so knows how to get to JFK but not to the New Otani Hotel. The airline routes passengers between major airports. The taxi driver in Osaka also knows only local destinations, so, when returning to New York, I tell the driver that my destination is “Osaka Airport,” not “New York.” Any single element of the transportation system knows only the _local_ geography. This leads to obvious efficiencies: the cabbie in New York needs to know only the New York metropolitan area, and the taxi driver in Osaka needs to know only the area in and around Osaka; the airline is the backbone linking JFK to Osaka.

Much like the transportation system just described, Open Shortest Path First (OSPF) is a _hierarchical_ routing protocol, implying that the IP network has a geography with each _area_ possessing only local routing information. In contrast, RIP and IGRP are _flat,_ implying that there is no hierarchy in the network -- every router possesses routes to every destination in the network. Right away, you can see that a flat routing protocol has inherent inefficiencies -- in our analogy, if the architecture of the transportation system was flat, the cabbie in New York would have to learn directions to the New Otani Hotel.

A hierarchical architecture, whether that of a transportation system or that of OSPF, allows the support of large systems because each area is responsible only for its local routes. RIP and IGRP cannot support very large networks because the routing overhead increases linearly with the size of the network.

Another radical difference from RIP and IGRP is that OSPF is not a DV protocol -- OSPF is based on a Link State algorithm, Dijkstra. What is a Link State algorithm? _Link_ refers to a router interface; in other words, the attached network. _State_ refers to characteristics of the link such as its IP address, subnet mask, cost (or metric), and operational status (up or down). Routers executing OSPF describe the state of their directly connected links in _link state advertisement_ (LSA) packets that are then flooded to all other routers. Using all the LSAs it receives, each router builds a topology of the network. The network topology is described mathematically in the form of a graph.

This topological database is the input to Dijkstra’s Shortest Path First (SPF) algorithm. With itself as the root, each router runs the SPF algorithm to compute the shortest path to each network in the graph. Each router then uses its shortest-path tree to build its routing table. Compare this with DV protocols: DV protocols propagate routes from router to router (this is sometimes called routing by rumor) and each router chooses the best route (to each destination) from all the routes (to that destination) that it hears.

DV protocols have to set up special mechanisms to guard against bad routing information that could propagate from router to router. In contrast, routers running the SPF algorithm need to ensure the accuracy of their LS databases; as long as each router has the correct topology information, it can use the SPF algorithm to find the shortest path.

Dijkstra’s algorithm is a wonderful tool but, as we shall see in more detail later, the SPF algorithm is expensive in terms of CPU utilization. The cost of running the algorithm increases quickly as the network topology grows. This would be a problem but, given OSPF’s hierarchical structure, the network is divided into “small” areas, and the SPF algorithm is executed by each router only on its intra-area topology. So how do routers in two different areas communicate with each other? All areas summarize their routes to a special area called the _backbone area_ or _area 0_. The backbone area in turn summarizes routes to all attached areas. Hence, traffic between any two areas must pass through the backbone area (see [Figure 6-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06.html#iprouting-CHP-6-FIG-1 "Figure 6-1. Overview of OSPF areas")).

![[Pasted image 20241110181931.png]]
Figure 6-1. Overview of OSPF areas

OSPF derives its name from Dijkstra’s SPF algorithm; the prefix “O” signifies that it’s an “open” protocol and so is described in an “open” book that everyone can access. That open book is RFC 2328, thanks to John Moy. In contrast, IGRP and EIGRP are Cisco _proprietary_ protocols. Multiple vendors support OSPF.

## Getting OSPF Running

Getting RIP, IGRP, and EIGRP running is easy, as we saw in earlier chapters. When TraderMary’s network grew to London, Shannon, Ottawa, etc., the DV routing protocols adapted easily to the additions. Getting OSPF running on a small network is also easy, as we will see in this chapter. However, unlike RIP, IGRP, and EIGRP, OSPF is a hierarchical protocol. OSPF does not work well if the network topology grows as a haphazard mesh.

In this section, we will configure OSPF on a small network. In later sections, we will learn how to build hierarchical OSPF networks.

TraderMary’s network, shown in [Figure 6-2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06.html#iprouting-CHP-6-FIG-2 "Figure 6-2. TraderMary’s network"), can be configured to run OSPF as follows.

![[Pasted image 20241110181957.png]]
Figure 6-2. TraderMary’s network

Like RIP and IGRP, OSPF is a distributed protocol that needs to be configured on every router in the network:

```
   hostname NewYork
   ...
   interface Ethernet0
   ip address 172.16.1.1 255.255.255.0
   !
   interface Serial0
   description New York to Chicago link
   ip address 172.16.250.1 255.255.255.0
   !
   interface Serial1
   description New York to Ames link
1  **`bandwidth 56`**                            
   ip address 172.16.251.1 255.255.255.0
   ...
   **`router ospf 10`**
   **`network 172.16.0.0 0.0.255.255 area 0`**
```

The **router ospf** command starts the OSPF process on the router. The syntax of this command is:

```
router ospf _`process-id`_
```

The _process-id_ , which should be between 1 and 65,535, is used to identify the instance of the OSPF process. The _process-id_ configured in the previous example is 10. Router _Chicago_ is similarly configured with the same _process-id_:

```
hostname Chicago
...
interface Ethernet0
ip address 172.16.50.1 255.255.255.0
!
interface Serial0
description Chicago to New York link
ip address 172.16.250.2 255.255.255.0
!
interface Serial1
description Chicago to Ames link
ip address 172.16.252.1 255.255.255.0
...

**`router ospf 10`**
**`network 172.16.0.0 0.0.255.255 area 0`**
```

Router _Ames_ is also configured with OSPF:

```
   hostname Ames
   ...
   interface Ethernet0
   ip address 172.16.100.1 255.255.255.0
   !
   interface Serial0
   description Ames to Chicago link
   ip address 172.16.252.2 255.255.255.0
   !
   interface Serial1
   description Ames to New York link
2  **`bandwidth 56`**                               
   ip address 172.16.251.2 255.255.255.0
   ...

   **`router ospf 10`**
   **`network 172.16.0.0 0.0.255.255 area 0`**
```

We next identify the networks that will be participating in the OSPF process and associate an area ID with each network. The syntax of this command is:

```
network _`address wildcard-mask`_ area _`area-id`_
```

The _address_ and _wildcard-mask_ fields identify a network by its IP address. Networks that match the _address_ and _wildcard-mask_ fields are associated with the area _area-id_. How is a network’s IP address matched against _address_ and _wildcard-mask_?

_wildcard-mask_ is a string of zeros and ones. An occurrence of a zero in _wildcard-mask_ implies that the IP address being checked must exactly match the corresponding bit in _address_. An occurrence of a one in _wildcard-mask_ implies that the corresponding bit in the IP address field is a “don’t care bit” -- the match is already successful.

Thus, the following clause can be read as stating that the first 16 bits of an IP address must be exactly “172.16” for the address to match the clause and be associated with area and that the next 16 bits of the IP address are “don’t care bits”:

```
network 172.16.0.0 0.0.255.255 area 0
```

Any IP address, such as `172.16.x.y`, will match this _address/wildcard-mask_ and be assigned the area ID of 0. Any other address, such as `10.9.x.y`, will not match this _address/wildcard-mask_.

If an interface IP address does not match the _address/wildcard-mask_ on a network statement, OSPF will check for a match against the next network statement, if there is another statement. Hence, the order of network statements is important. If an interface IP address does not match the _address/wildcard-mask_ on any network statement, that interface will not participate in OSPF.

There is more than one method of assigning area IDs to networks. The most rigorous method specifically lists every network when making a match. The wildcard mask contains only zeros:

```
hostname NewYork
...
**`router ospf 10`**
            
**`network 172.16.1.1 0.0.0.0 area 0`**
            
**`network 172.16.250.1 0.0.0.0 area 0`**
            
**`network 172.16.251.1 0.0.0.0 area 0`**
```

The most loose method is an all-ones wildcard mask:

```
hostname NewYork
...
**`router ospf 10`**
            
**`network 0.0.0.0 255.255.255.255 area 0`**
```

Note that in the second (loose) method, network 192.168.1.1 also belongs to area 0.

If an IP address does not match an area-ID specification, the match continues to the next statement. So, for example, a router may be configured as follows:

```
network 172.16.0.0 0.0.255.255 area 0
network 192.0.0.0 0.255.255.255 area 1
```

An IP address of `192.168.1.1` will not match the first statement. The match will then continue to the next statement. All IP addresses with “192” in the first 8 bits will match the second clause and hence will fall into area 1. A network with the address `10.9.1.1` will not match either statement and hence will not participate in OSPF.

The _area-id_ field is 32 bits in length. You can specify the area ID in the decimal number system, as we did earlier, or in the dotted-decimal notation that we use for expressing IP addresses. Thus, the area ID `0.0.0.0` (in dotted decimal) is identical to the area ID (in decimal); the area ID `0.0.0.100` (in dotted decimal) is identical to 100 (in decimal); and the area ID `0.0.1.0` (in dotted decimal) is identical to 256 (in decimal). The area ID of is reserved for the backbone area. The area ID for nonbackbone areas can be in the range 1 to 4,294,967,295 (or, equivalently, `0.0.0.1` to `255.255.255.255`).

The **show ip ospf interface** command shows the assignment of area IDs to network interfaces:

```
   NewYork#sh ip ospf interface
   ...
   Ethernet0 is up, line protocol is up 
3    **`Internet Address 172.16.1.1/24, Area 0`** 
4    **`Process ID 10, Router ID 172.16.251.1, Network Type BROADCAST, Cost: 10`**
     ...
   Serial0 is up, line protocol is up 
     **`Internet Address 172.16.250.1/24, Area 0`** 
     Process ID 10, Router ID 172.16.251.1, Network Type POINT_TO_POINT, Cost: 64 
   ...
   Serial1 is up, line protocol is up 
     **`Internet Address 172.16.251.1/24, Area 0`** 

     Process ID 10, Router ID 172.16.251.1, Network Type POINT_TO_POINT, Cost: 1785  
   ...
```

The routing tables for _NewYork_, _Chicago_, and _Ames_ will show all `172.16.0.0` subnets. Here is _NewYork_’s table:

```
   NewYork#sh ip route
   Codes: C - connected, S - static, I - IGRP, R - RIP, M - mobile, B - BGP
          D - EIGRP, EX - EIGRP external, O - OSPF, IA - OSPF inter area 
          N1 - OSPF NSSA external type 1, N2 - OSPF NSSA external type 2
          E1 - OSPF external type 1, E2 - OSPF external type 2, E - EGP
          i - IS-IS, L1 - IS-IS level-1, L2 - IS-IS level-2, * - candidate default
       
   Gateway of last resort is not set

5       172.16.0.0/16 is variably subnetted, 6 subnets, 2 masks      
6  O       172.16.252.0/24 [110/128] via 172.16.250.2, 01:50:18, Serial0
   C       172.16.250.0/24 is directly connected, Serial0
   C       172.16.251.0/24 is directly connected, Serial1
7  O       172.16.50.1/32 [110/74] via 172.16.250.2, 01:50:18, Serial0   
   C       172.16.1.0/24 is directly connected, Ethernet0
8  O       172.16.100.1/32 [110/138] via 172.16.250.2, 01:50:18, Serial0
```

The OSPF-derived routes in this table are labeled with an “O” in the left margin. Note that the routing table provides summary information (as in line 5). This line contains subnet mask information (24 bits, or `255.255.255.0`) and the number of subnets in `172.16.0.0` (6).

## OSPF Metric

Each OSPF router executes Dijkstra’s SPF algorithm to compute the shortest-path tree from itself to every subnetwork in its area. However, RFC 2328 does not specify how a router should compute the cost of an attached network -- this is left to the vendor. Cisco computes the cost of an attached network as follows:

_Cost =_ 10^8/_bandwidth of interface in bits per second_

Using this definition, the OSPF cost for some common media types is shown in [Table 6-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s02.html#iprouting-CHP-6-TABLE-1 "Table 6-1. Default OSPF costs"). [Table 6-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s02.html#iprouting-CHP-6-TABLE-1 "Table 6-1. Default OSPF costs") assumes default interface bandwidth. Note that the cost is rounded down to the nearest integer.

Table 6-1. Default OSPF costs

| Media type                                                                                                                   | Default bandwidth | Default OSPF cost |
| --------------------------------------------------------------------------------------------------------------------------- | ----------------- | ----------------- |
| Ethern                                                                                                                       | 10 Mbps           | 10                |
| Fast Ether                                                                                                                   | 100 Mbps          | 1                 |
|                                                                                                                              | 100 Mbps          | 1                 |
| T-1 (serial interface)[[a](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s02.html#ftn.ch06-FTNOT       | 1,544 kbps        | 64                |
| 56 kbps (serial inte                                                                                                         | 1,544 kbps        | 64                |                                                                                                                                | 45,045 kbps       | 2                                                                                                                                         ameters. |                   |                   |
All serial interfaces on Cisco routers are configured with the same default bandwidth (1,544 kbits/s) and delay (20,000 ms) parameters.

The OSPF cost computed by a router can be checked with the command **show ip ospf interface** , as in line 4 in the code block in the previous section, where the cost of the Ethernet segment is 10. The composite cost of reaching a destination is the sum of the individual costs of all networks in the path to the destination and can be seen as output of the **show ip route** command in lines 6, 7, and 8.

The default value of the OSPF metric may not be adequate in some situations. For example, in TraderMary’s configuration, the _NewYork_ → _Ames_ link runs at 56 kbps, but the default metric makes it appear to be a T-1. This was fixed by modifying the interface bandwidth, as in lines 3 and 4 in the previous section. The command to modify a bandwidth is:

```
bandwidth _`kilobits`_
```

Keep in mind that modifying the interface bandwidth impacts other protocols that utilize the bandwidth parameter, such as IGRP. Modifying bandwidth may not always be viable. In such situations, the OSPF cost of an interface may be directly specified:

```
ip ospf cost _`value`_
```

where _value_ is an integer in the range 1 to 65,535 (OSPF sets aside two octets to represent interface cost, as we will see later in the section [Section 6.4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html "How OSPF Works")).

This approach to calculating OSPF costs does not work well for network speeds greater than 100 Mbps. The OSPF cost for all speeds greater than the reference bandwidth is rounded up to 1, and there is no way to distinguish between one network and another. The network engineer has two approaches to choose from here. First, manually configure the OSPF cost for all interfaces equal to or faster than 100 Mbps. For example, all FE interfaces may be configured with a cost of 8, OC-3 interfaces with a cost of 6, and GE interfaces with a cost of 4. Second, redefine the reference bandwidth with the following command:

```
ospf auto-cost reference-bandwidth _`reference-bandwidth`_
```

where _reference-bandwidth_ is in Mbps. When this command is used, the cost of an interface is calculated as:

_Cost = reference-bandwidth-in-bps/bandwidth of interface in bits per second_

This command is available in Cisco IOS Releases 11.2 and later. If the reference bandwidth is modified, it must be modified on all routers in the OSPF domain. The default value of _reference-bandwidth_ is 108.

The developers of OSPF envisaged (as an optional feature) multiple _types of service_ (TOS) with differing metrics for each TOS. Using this concept, bulk data may be routed, say, over a satellite link, whereas interactive data may be routed under the sea. However, the TOS concept has not been carried into any major implementations -- Cisco supports only one TOS.

## Definitions and Concepts

Dijkstra’s algorithm solves the problem of discovering the shortest path from a single source to all vertices in a graph where the edges are each represented with a cost. For example, a car driver could use Dijkstra’s algorithm to find the shortest paths from New York to major cities in the northeastern U.S. and Canada. The input to Dijkstra would be a graph that could be represented by a matrix like that shown in [Table 6-2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-TABLE-2 "Table 6-2. Driving distances").

Table 6-2. Driving distances

|Town name|Town name|Driving distance (miles)|
|---|---|---|
|New York|Washington|236|
|New York|Boston|194|
|Boston|Chicago|996|
|Washington|Chicago|701|
|New York|Toronto|496|
|Detroit|Chicago|288|
|Washington|Detroit|527|
|Boston|Toronto|555|
|Toronto|Detroit|292|

The output would be the shortest paths from New York to all other cities in the graph. A geographical view of [Table 6-2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-TABLE-2 "Table 6-2. Driving distances") is contained in [Figure 6-3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-FIG-3 "Figure 6-3. Geographical view of driving distances").

![[Pasted image 20241110182537.png]]
Figure 6-3. Geographical view of driving distances

There are six nodes in this graph: New York, Chicago, Boston, Toronto, Detroit, and Washington. There are nine edges in the graph, each represented by the distance between a pair of vertices. The SPF algorithm works as follows:

1. Starting at the source node -- New York -- build a list of one-segment paths originating at the source node. This list will be New York → Washington, New York → Boston, and New York → Toronto.
2. Sort this list in increasing order. The sorted list will be New York → Boston (194), New York → Washington (236), and New York → Toronto (496).
3. Pick the shortest path from this list -- New York → Boston -- and move Boston to the list of vertices for which the shortest path has been identified.
4. Next, append a new list of paths to the list that was defined in step 1. The list to be appended consists of one-segment paths starting from Boston. This list will be Boston → Chicago and Boston → Toronto. The composite list will be New York → Washington, New York → Toronto, Boston → Chicago, and Boston → Toronto.

The algorithm continues, as in step 2, and the composite list is sorted in increasing order with distances from the source node: New York → Washington (236), New York → Toronto (496), New York → Boston → Toronto (194 + 555 = 749), and New York → Boston → Chicago (194 + 996 = 1,190). In step 3, the shortest path is again picked from the top of the list and Washington is added to the list of vertices for which the shortest path has been identified. The algorithm continues until the shortest paths to all cities have been identified.

OSPF employs Dijkstra’s SPF algorithm to compute the shortest path from a router to every network in the graph. In OSPF terminology, this graph of the network topology (similar to [Table 6-2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-TABLE-2 "Table 6-2. Driving distances")) is referred to as the topological database or the _link state database_ . Each router executes the SPF algorithm with itself as the source node. The results of the SPF algorithm are the shortest paths to each IP network from the source node; hence, this constitutes the IP routing table for the router.

Although the database of [Table 6-2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-TABLE-2 "Table 6-2. Driving distances") is relatively static -- driving distances change only when new roads are built or old roads are closed -- the LS database for a network is quite dynamic because of changes in the state of subnetworks. A link may go down or come up. A network administrator may make changes to the status of a link, such as shutting it down or changing its cost. Every time there is any change in a router’s LS database, Dijkstra’s SPF algorithm needs to be run again. It can be shown that the SPF algorithm takes ElogE time to run, where E is the number of edges in the graph.

As the size of a network grows, Dijkstra will consume more and more memory and CPU resources at each router. In other words, Dijkstra does not scale for large topologies. Fortunately, OSPF has a clever solution to this problem: break the network into _areas_ and execute Dijkstra only on each _intra-area_ topology.

An area is a collection of _contiguous_ networks and routers that share a unique area ID. Each area maintains its own topological database: other areas do not see this topological information. The SPF algorithm is executed on each intra-area topology by the intra-area routers.

Containing the number of routers and networks in an area allows OSPF to scale to support large networks. The network can grow almost without bounds with the addition of new areas. If a single area becomes too large, it can be split into two or more areas.

Before a router can execute the SPF algorithm, it must have the most recent topological database for its area(s). Note the plural: a router may have interfaces in multiple areas. A topological change in an area will cause SPF to recompute on all routers with interfaces in that area. Routers in other areas will not be affected by the change. Breaking a network into areas is thus akin to breaking a network into smaller, independent networks.

Unlike flat networks such as RIP and IGRP in which each router has the same responsibilities and tasks, OSPF’s hierarchy imposes a structure in which routers and even areas are differentiated with respect to their roles.

### Backbone Area

The _backbone area_ is of special significance in OSPF because all other areas must connect to it. The area ID of (or `0.0.0.0`) is reserved for the backbone. [Figure 6-4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-FIG-4 "Figure 6-4. OSPF architecture: a high-level view") shows an OSPF network comprised of a backbone area and three other areas -- areas 1, 2, and 3. Note that all inter-area traffic must pass through the backbone area, which implies that backbone routers must possess the complete topological database for the network.

![[Pasted image 20241110182638.png]]
Figure 6-4. OSPF architecture: a high-level view

### Backbone Router

A router with an interface in area is referred to as a _backbone router_ . A backbone router may also have interfaces in other areas. Routers _R1_, _R2_, _R3_, _R4_, and _R5_ in [Figure 6-4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-FIG-4 "Figure 6-4. OSPF architecture: a high-level view") are backbone routers.

The backbone routers hold a topological database that describes the state of all backbone links_,_ summary links describing IP networks in areas 1, 2, and 3, and external links that describe the IP network in the RIP network.

### Area or Regular Area

A _regular area_ has a unique area ID in the range 1 (or `0.0.0.1`) to 4,294,967,295 (`255.255.255.255`).

A router in, say, area 1 will hold topological information for the state of all area 1 links, summary links that describe IP networks in areas 0, 2, and 3, and external links that describe IP networks in those networks.

### Internal Router

An _internal router_ has interfaces in one area only. Routers _R6_, _R7_, and _R8_ in [Figure 6-4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-FIG-4 "Figure 6-4. OSPF architecture: a high-level view") are internal routers in area 1.

### Area Border Router (ABR)

An _area border router_ has interfaces in more than one area. Routers _R3_, _R4_, and _R5_ in [Figure 6-4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-FIG-4 "Figure 6-4. OSPF architecture: a high-level view") are ABRs.

An ABR has topological information for multiple areas. Router _R3_ is an ABR that holds topological databases for areas and 1. Router _R4_ holds topological databases for areas 0, 2, and 3. Router _R5_ holds topological databases for areas and 3.

An ABR can summarize the topological database for one of its areas. Router _R3_ may summarize the topological database for area 1 into area 0. Summarization is key in reducing the computational complexity of the OSPF process.

### Autonomous System Boundary Router (ASBR)

An _autonomous system boundary router_ imports routing information from another AS into OSPF. The routes imported into OSPF from the other AS are referred to as _external routes_ .

Router _R9_ in [Figure 6-4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-FIG-4 "Figure 6-4. OSPF architecture: a high-level view") is an ASBR. _R9_ imports RIP routes from an external network into OSPF. An ASBR may be configured to summarize external routes into OSPF.

### Stub Area

Consider an area with no direct connections to any external networks. Importing external records into this area may be unnecessary because all traffic to external networks must be routed to the ABRs. Such an area can use a default route (in place of external routes) to send all external IP traffic to its ABRs.

Configuring an area as a _stub area_ blocks the advertisement of external IP records at the ABRs and instead causes the ABRs to generate default routes into the stub area.

Routers in a stub area hold a topological database that describes the state of all local links, summary links describing IP networks in other areas, but no external networks. This reduction in the size of the topological database saves on processor and memory resources. A stub area may use routers with less memory/CPU power or use the spare memory/CPU resources to build a _large_ stub area.

There is a potential disadvantage to configuring an area as a stub area. For example, if area 3 in [Figure 6-4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-FIG-4 "Figure 6-4. OSPF architecture: a high-level view") is configured as a stub area, _R4_ and _R5_ will each advertise a default route into the stub area. An external route may be closer to _R4_, but routers in the stub area will lose that information and route all external traffic to _R4_ or _R5_, depending on which one is closer. Stub areas cannot support external connections since stub routers do not carry external LSAs. Stub areas cannot support virtual links, which I’ll discuss later in this chapter, for similar reasons.

### Totally Stubby Area

A _totally stubby area_ carries the concept of a stub area further by blocking summary records for IP networks in other areas at the ABRs. All inter-area and external traffic is matched to the default route announced by the ABR(s).

In terms of LSA types, routers in totally stubby areas hold a topological database that describes the state of all local links only.

Just like a stub area, a totally stubby area cannot support connections to external networks.

### Not So Stubby Area (NSSA)

_Not so stubby areas_ are stub areas with one less restriction: NSSAs can support external connections. In all other respects, NSSAs are just like stub areas -- routers in NSSAs do not carry external LSAs, nor do they support virtual links.

Any area that can be configured as a stub area but needs to support an external network can be changed into an NSSA.

### OSPF Topological Database

The OSPF topological database is composed of link state advertisements (LSAs). OSPF routers originate LSAs describing a piece of the network topology; these LSAs are flooded to other routers that then compose a database of LSAs. There are several types of LSAs, each originating at a different router and describing a different component of the network topology. The various types of LSAs are:

Router LSA (type 1)
	A router LSA describes a router’s links (or interfaces). All routers originate router LSAs. A router LSA is flooded to all intra-area routers.

Network LSA (type 2)
	A network LSA describes a broadcast network (such as an Ethernet segment) or a non-broadcast multi-access (NBMA) network (such as Frame Relay). All routers attached to the broadcast/NBMA network are described in the LSA. A network LSA is flooded to all intra-area routers.

Summary LSA (type 3)
	A summary LSA describes IP networks in another area. The summary LSA is originated by an ABR and flooded outside the area. Summary LSAs are flooded to routers in all OSPF areas except totally stubby areas.

ASBR summary LSA (type 4)
	ASBR summary LSAs describe the route to an ASBR. The mask associated with these LSAs is 32 bits long because the route they advertise is to a host -- the IP address of the ASBR. ASBR summary LSAs originate at ASBRs. ASBR summary LSAs are flooded to routers in all OSPF areas except stub areas.

External LSA (type 5)
	External LSAs describe routes external to the OSPF process (in another autonomous system). An external route can be a default route. External LSAs originate at the ASBR. External LSAs are flooded throughout the OSPF network, except to stub areas.

NSSA external LSA (type 7)
	NSSA external LSAs describe routes to external networks (in another autonomous system) connected to the NSSA. Unlike type 5 external LSAs, NSSA external LSAs are flooded only within the NSSA. Optionally, type 7 LSAs may be translated to type 5 LSAs at the ABR and flooded as type 5 LSAs.

### OSPF Route Types

Every router in OSPF uses its local topological database as input to the SPF algorithm. The SPF algorithm yields the shortest path to every known destination, which is then used to populate the IP routing table as one of four route types:

Intra-area route
	An intra-area route describes the route to a destination within the area.

Inter-area route
	An inter-area route describes the route to a destination in another area. The path to the destination comprises an intra-area path, a path through the backbone area and an intra-area path in the destination network’s area. An inter-area route is sometimes referred to as a summary route.

External route (type 1)
	An external route describes the route to a destination outside the AS. The cost of a type 1 external route is the sum of the costs of reaching the destination in the external network and the cost of reaching the ASBR advertising the route.

External route (type 2)
	An external route describes the route to a destination outside the AS. The cost of a type 2 external route is the cost of reaching the destination in the external network only; it does not include the cost of reaching the ASBR advertising the route.

When routing a packet, the routing table is scanned for the most specific match. For example, say that the destination IP address in the packet is `10.1.1.254` and the routing table contains entries for `10.1.1.0/24` and `10.1.1.192/26`. The most specific match will be the route `10.1.1.192/26`. Now, what if `10.1.1.192/26` was known as an intra-area route and an inter-area route? OSPF prefers routes in the following order: intra-area routes (most preferred), inter-area routes, type 1 external routes, and type 2 external routes (least preferred).

Note the order in which the rules were applied: first the route with the most specific match was identified and then the OSPF preferences were applied. Thus, when routing the packet with the destination address `10.1.1.254`, if the routing table shows `10.1.1.0/24` as an intra-area route and `10.1.1.192/26` as a type 2 external route, the most specific match (`10.1.1.192/26`) will win. If OSPF has multiple equal-cost routes to a destination, it will load-balance traffic over those routes.

## How OSPF Works

OSPF routers must first discover each other before they can exchange their topological databases. Once each router has the complete topological database, it can use the SPF algorithm to compute the shortest path to every network. This section focuses on neighbor discovery and the exchange of topological databases.

Let’s begin at the beginning. OSPF packets are encapsulated directly in IP with the protocol field set to 89. The destination IP address in OSPF depends on the network type. OSPF uses two IP multicast addresses on broadcast and point-to-point networks: `225.0.0.5` for all OSPF routers and `224.0.0.6` for all DR/BDR (designated router/backup designated router) routers. Using IP multicast addresses is more efficient than using broadcast addresses. If broadcast addresses are used, all attached devices must receive the broadcast packet, unwrap it, and then discard the contents if they are not running OSPF. NBMA networks and virtual links use unicast addresses because they do not support multicast addresses.

Following the IP header is the OSPF header (see [Figure 6-5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-FIG-5 "Figure 6-5. Format of an OSPF header")). The OSPF header is common to all types of OSPF packets. The following list defines the format of the OSPF header and the five types of OSPF packets:

_Version_
The OSPF version in use. The current version number is 2.

_Type_
There are five types of OSPF packets:
	Type 1 : Hello packets, described in the next section.
	Type 2 : Database description packets, described later under [Section 6.4.5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-SECT-4.5 "Database Exchange").
	Type 3 : Link state requests, described in [Section 6.4.5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-SECT-4.5 "Database Exchange").
	Type 4 : Link state updates, described in [Section 6.4.5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-SECT-4.5 "Database Exchange").
	Type 5 : Link state acknowledgments, described in [Section 6.4.5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-SECT-4.5 "Database Exchange").

_Packet length_
The length of the OSPF packet, including the header.

_Router ID_
The router ID of the router originating the OSPF packet.

_Area ID_
The area ID of the network on which this packet is being sent.

_Checksum_
The checksum for the entire packet, including the header.

_Au type_
The type of authentication scheme in use. The possible values for this field are:
	0 : No authentication
	1 : Clear-text password authentication
	2 : MD5 checksum

_Authentication data_
The authentication data.

![[Pasted image 20241110183137.png]]
Figure 6-5. Format of an OSPF header

### Neighbor Discovery: The Hello Protocol

Every router generates OSPF hello packets on every OSPF-enabled interface. Hello packets are sent every 10 seconds on broadcast media and every 30 seconds on nonbroadcast media. Routers discover their neighbors by listening to hellos. The output of the command **show ip ospf neighbor** lists the neighbors that have been discovered.

Each hello packet contains the fields described in the following sections. The format of a hello packet is shown in [Figure 6-6](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-FIG-6 "Figure 6-6. Format of hello packet").

![[Pasted image 20241110183201.png]]
Figure 6-6. Format of hello packet

#### Router ID

When the OSPF process first starts on a router (e.g., when the router is powered up) it attempts to establish a _router ID._ The router ID is the name or label that will be attached to the node representing the router in the SPF topology graph. If OSPF cannot establish a router ID, the OSPF process aborts.

How does a router choose its router ID? There are two situations to consider here:

- If a router has one or more loopback interfaces, it chooses the highest IP address from the pool of loopback interfaces as its router ID. Loopback interfaces are always active.
- If a router has no loopback interfaces, it chooses the highest IP address from any of its active interfaces as its router ID. If a router has no active interface with an IP address, it will not start the OSPF process.

The router ID is chosen when the OSPF process first starts: the addition or deletion of interfaces or addresses on a router after the router ID has been selected does not change the router ID. A new router ID is picked only when the router is restarted (or when the OSPF process is restarted).

So, for example, the router ID of _NewYork_ can be checked as follows:

```
NewYork#sh ip ospf 
 Routing Process "ospf 10" with ID **`172.16.251.1`**
 Supports only single TOS(TOS0) routes
 SPF schedule delay 5 secs, Hold time between two SPFs 10 secs
......
```

In this example, the router ID was derived using the router’s highest IP address. It is usually preferable to configure loopback interfaces to assign predictable router IDs to OSPF routers (since a loopback interface is a virtual interface and will not go down, as a physical interface would). The router ID must be unique within the topology database.

The configuration on _NewYork_ may be modified as follows:

```
hostname NewYork
!
interface Loopback0
 ip address 192.168.1.1 255.255.255.255
...
```

After _NewYork_ is rebooted, its router ID will change as follows:

```
NewYork#sh ip ospf
 Routing Process "ospf 10" with ID 192.168.1.1
...
```

Since the router ID is critical to the OSPF process, it is important for the network engineer to maintain a table of all router IDs.

Note the following points:

1. Since the router ID is needed only to represent the router in the SPF graph, it is not required that OSPF advertise the router ID. However, if the router ID is advertised, it will be represented as a stub link in a router LSA.
    
2. A mask of `255.255.255.255` may be chosen for the loopback interface to conserve on network addresses, as in the earlier example.
    
3. If the router ID is not advertised, any unique address can be used to represent the router ID -- the use of nonreserved IP addresses will not cause any routing-table conflicts.
    

#### Area ID
The area ID of the interface on which the OSPF packet is being sent.

#### Checksum
The checksum pertaining to the hello packet.

#### Authentication
The authentication method and authentication data.

#### Network mask
The network mask of the interface on which the hello packet is being sent.

#### Hello-interval
The duration between hello packets. The default value of hello-interval is 10 seconds on most interfaces.

The hello-interval can be modified with the following command in interface configuration mode:

```
ip ospf hello-interval _`seconds`_
```
#### Options

OSPF defines several optional capabilities that a router may or may not support. The options field is one octet long, as shown in [Figure 6-7](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-FIG-7 "Figure 6-7. Format of the options field").

![[Pasted image 20241110183339.png]]
Figure 6-7. Format of the options field

Routers that support demand circuits set the DC bit; NSSA support is signified using the N bit. The E bit signifies that the router accepts external LSAs -- stub routers turn off this bit. The T bit signifies the support of multiple types of service.

#### Router priority
A router with a higher priority takes precedence in the DR election algorithm. A value of makes the router ineligible for DR/BDR election. The default value of this field is 1.

#### Router dead-interval
If no hello packets are received for the duration of the dead-interval_,_ the neighbor is declared dead. This value can be altered with the following command in interface configuration mode:

```
ip ospf dead-interval _`value`_
```
#### Designated router (DR)

The designated router for multi-access networks. This field is set to `0.0.0.0` if no DR has been elected on the network.

#### Backup designated router

The IP address of the backup designated router’s interface on this network.This field is set to `0.0.0.0` if no BDR has been elected on the network.

#### Neighbor router ID list

The neighbor router ID list is the list of neighboring routers from which this router has received hellos within the last dead-interval seconds. Before a router lists its neighbor in its hello packet, the two routers must agree on the following: area ID, authentication mechanism, network mask, hello-interval, router dead-interval, and options fields. If these values match, the routers become neighbors and start listing each other in their hello packets.

The following output shows _NewYork_’s neighbors:

```
NewYork#show ip ospf neighbor

Neighbor ID     Pri   State           Dead Time   Address         Interface
192.168.1.2      1   **`FULL/`**  -        00:00:31    172.16.250.2    Serial0
192.168.1.3      1   **`FULL/`**  -        00:00:32    172.16.251.2    Serial1
```

Note that the state of _NewYork_’s relationship with both neighbors is “Full,” implying that the neighbors have exchanged LS databases to become adjacent. Under normal, stable conditions, the state of each neighbor relationship should be “2-way” or “Full.” “2-way” implies that the neighbors have seen each other’s hello packets but have not exchanged LSAs. In the process of maturing into a “Full” relationship, neighbors transition through the states “Exstart,” “Exchange,” and “Loading,” indicating that neighbors have seen each other’s hello packets and are attempting to exchange their LS databases. These are transitory states, all being well.

Then there are the problem states. “Down” indicates that a hello packet has not been received from the neighbor in the last router dead-interval. “Attempt” applies to NBMA networks and indicates that a hello has not yet been received from the neighbor. “Init” implies that a hello was received from the neighbor but its neighbor router ID list did not include the router ID of this router.

### DR/BDR Election

Consider _n_ routers on a broadcast network (such as Ethernet). If a router exchanged its topological database with every other router on the network, (_n_ x (_n_ - 1)) / 2 adjacencies would be formed on the segment. This would create a lot of OSPF overhead traffic. OSPF solves this problem by electing a _designated router_ (DR) and a _backup designated router_ (BDR) on each broadcast network. Each router on a broadcast network establishes an adjacency with only the DR and the BDR. The DR and the BDR flood this topology information to all other routers on the segment.

DR/BDR election can be described in the following steps. Remember that the DR/BDR election process occurs on every multi-access network (not router). A router may be the DR on one interface but not another.

The following description assumes that a router _R_ has just been turned up on a multi-access network:

1. On becoming active on a multi-access network, the OSPF process on router _R_ begins receiving hellos from neighbors on its interface to the multi-access network. If the hellos indicate that there already are a DR and a BDR, the DR/BDR election process is terminated (even if _R_’s OSPF priority is higher than the current DR/BDR priority).
    
2. If hellos from neighbors indicate that there is no active BDR on the network, the router with the highest priority is elected the BDR. If the highest priority is shared by more than one router, the router with the highest router ID wins.
    
3. If there is no active DR on the network, the BDR is promoted to DR.
    

The following can be stated as corollaries of the above rules:

1. If a DR and BDR have already been elected, bringing up a new router (even with a higher priority) will not alter the identities of the DR/BDR.
    
2. If there is only one DR-eligible router on a multi-access network, that router will become the DR.
    
3. If there are only two DR-elegible routers on a multi-access network, one will be the DR and the other, the BDR.
    

A router with a higher priority takes precedence during DR election. A priority value of indicates that the router is ineligible for DR election. The default priority value is 1. Routers with low memory and CPU resources should be made ineligible for DR election.

The router interface priority may be modified with the following command in interface configuration mode:

```
ip ospf priority _`number`_
```

where _number_ is between 0 and 255.

The state of an OSPF interface (including the result of the DR/BDR election process) can be seen as output of the **show ip ospf interface** command:

```
   NewYork#sh ip ospf interface
   Ethernet0 is up, line protocol is up 
     Internet Address 172.16.1.1/24, Area 0 
     Process ID 10, Router ID 172.16.251.1, Network Type BROADCAST, Cost: 10    
9    **`Transmit Delay is 1 sec, State DR, Priority 1`**         
     Designated Router (ID) 172.16.251.1, Interface address 172.16.1.1     
10   **`No backup designated router on this network`**                   
     Timer intervals configured, Hello 10, Dead 40, Wait 40, Retransmit 5 
       Hello due in 00:00:02                             
11   **`Neighbor Count is 0, Adjacent neighbor count is 0`**             
     Suppress hello for 0 neighbor(s)
   ...
```

Note that _NewYork_ is the DR on _Ethernet0_. Since there is no other router on this network, there is no BDR (line 10) and the router has not established any adjacencies (line 11).

### Interface State

The state of an interface can have one of the following values:

Down
The interface state is down as indicated by lower-level protocols, and no OSPF traffic has been sent or received yet.

Loopback
The interface is looped and will be advertised in LSAs as a host route.

Point-to-point
The interface is up and is recognized as a serial interface or a virtual link. After entering the point-to-point state, the neighbors will attempt to establish adjacency.

Waiting
This state applies only to broadcast/NBMA networks on which the router is attempting to identify the DR/BDR.

DR
This router is the DR on the attached network.

Backup
This router is the BDR on the attached network.

DRother
This router is neither the DR nor the BDR on the attached network. The router will form adjacencies with the DR and BDR (if they exist).

As an example, the state of _NewYork_’s interface to _Chicago_ is point-to-point (line 12) and _NewYork_ and _Chicago_ have established adjacency (lines 13 and 14):

```
    NewYork#sh ip ospf interface
    ...
    Serial0 is up, line protocol is up 
      Internet Address 172.16.250.1/24, Area 0 
      Process ID 10, Router ID 172.16.251.1, Network Type POINT_TO_POINT, Cost: 64 
12    **`Transmit Delay is 1 sec, State POINT_TO_POINT,`** 
      Timer intervals configured, Hello 10, Dead 40, Wait 40, Retransmit 5 
        Hello due in 00:00:01                             
13    **`Neighbor Count is 1, Adjacent neighbor count is 1`**
14      **`Adjacent with neighbor 69.1.1.1`** 
      Suppress hello for 0 neighbor(s)
```
### Neighbor Relationship

Not all neighbors establish adjacency. Neighbors may stay at “2-way” or enter into a “Full” relationship, depending on the type of network, as follows:

Point-to-point networks
Routers on point-to-point networks always establish adjacency.

Broadcast networks
Routers on broadcast networks establish adjacency only with the DR and the BDR, maintaining a 2-way relationship with the other routers on the network.

Non-broadcast multi-access (NBMA) networks
Routers on NBMA networks establish adjacency only with the DR and the BDR.

Virtual links
Routers on virtual links always establish adjacency.

### Database Exchange

The _database description (DD) packet_ is used to describe the contents of the LS database to a peer OSPF router. Only LSA headers are sent in DD packets; the peer router responds by sending its own LSA headers in DD packets.

The LSA header ([Figure 6-8](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-FIG-8 "Figure 6-8. Format of an LSA header")) uniquely identifies a piece of the OSPF network topology. The key fields in the LSA header are the _advertising router_ , _LS type_ , and _link state ID_ . The advertising router is the router ID of the originator of the LSA. The LS type identifies the type of the LSA that follows. The link state ID depends on the LS type, as shown in [Table 6-3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-TABLE-3 "Table 6-3. LS type and link state ID").

![[Pasted image 20241110183651.png]]
Figure 6-8. Format of an LSA header

Table 6-3. LS type and link state ID

|LS type|Link state ID|
|---|---|
|1 (router LSA)|Router ID of the originator of the LSA|
|2 (network LSA)|IP address of the DR’s interface to the multi-access network|
|3 (summary LSA)|IP address of the destination network|
|4 (ASBR summary LSA)|Router ID of the ASBR|
|5 (external LSA)|IP address of the destination network|

Several copies of an LSA may be circulating in a network. The _LS sequence number_,a signed 32-bit integer, helps identify the most recent LSA. The first instance of an LSA record contains a sequence number field of 0x80000001. Each new instance of the LSA contains a sequence number that is one higher. The maximum sequence number is 0x7fffffff, after which the sequence numbers are recycled. The sequence number helps identify the most recent instance of an LSA.

Upon receiving LSA headers in DD packets, both routers check to see if this piece of the OSPF topology is already contained in their LS databases. In this process, the advertising router, LS type, and link state ID fields (from the LSA header) are compared against the router’s LS database. If no matching records are found or if a matching record is found with a lower sequence number, the complete LSA is requested using the _link state request_ _packet_. The LS request packet contains the LSA header to help identify the record being sought.

In response to a link state request, a router issues a link state update containing the LSA. The LSA completely describes the piece of OSPF topology in question. LS updates are issued (a) in response to an LS request, as just described; (b) because of a change in the state of the link; and (c) every 30 minutes, with a new sequence number and the age field set to 0.

All LS updates are acknowledged in _link state acknowledgment packets_ (see [Figure 6-9](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-FIG-9 "Figure 6-9. Database description, link state request, link state update, and link state acknowledgment packets")).

![[Pasted image 20241110183753.png]]
Figure 6-9. Database description, link state request, link state update, and link state acknowledgment packets

There are six types of LSA records, each representing a different piece of the network topology. We’ll use TraderMary’s network with a French extension ([Figure 6-10](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-FIG-10 "Figure 6-10. TraderMary’s network with a French extension")) to take a closer look at the various LSA types.

![[Pasted image 20241110183827.png]]
Figure 6-10. TraderMary’s network with a French extension

TraderMary’s network in New York is configured as follows. _NewYork2_ is an ABR with a serial link in area 1 to router _Paris_ (line 15).

```
    hostname NewYork2
    !
    interface Loopback0
     ip address 192.168.1.4 255.255.255.0
    !
    interface Ethernet0
     ip address 172.16.1.2 255.255.255.0
     ip pim sparse-mode
    !
    interface Serial1
     description Paris link
     ip address 10.0.1.2 255.255.255.0
     bandwidth 56
    !
    router ospf 10
     network 172.16.0.0 0.0.255.255 area 0
15   **`network 10.0.0.0 0.255.255.255 area 1`** 
```

_Paris_ is an ASBR redistributing RIP routes from a legacy network into OSPF (line 16):

```
    hostname Paris
    !
    interface Loopback0
     ip address 192.168.1.5 255.255.255.255
    !
    interface Ethernet0
     ip address 10.0.2.1 255.255.255.0
    !
    interface Serial1
     description link to NewYork2
     ip address 10.0.1.1 255.255.255.0
    !
    router ospf 10
16   **`redistribute rip metric 100 subnets`**     
     network 10.0.0.0 0.255.255.255 area 1
    !
    router rip
     network 10.0.0.0
```

The `10.0.0.0` subnets -- `10.0.1.0`, `10.0.2.0`, and `10.0.3.0` -- are known to both the OSPF and RIP processes on router _Paris_. Let’s see how _NewYork_ learns these subnets. Here is _NewYork_’s routing table:

```
    NewYork#sh ip route
    ...
         10.0.0.0/24 is subnetted, 3 subnets
17  **`O IA    10.0.2.0 [110/1805] via 172.16.1.2, 00:07:45, Ethernet0`**
18  **`O E2    10.0.3.0 [110/100] via 172.16.1.2, 00:07:46, Ethernet0`**
19  **`O IA    10.0.1.0 [110/1795] via 172.16.1.2, 00:07:46, Ethernet0`**  
         192.168.1.0/32 is subnetted, 1 subnets
    C       192.168.1.1 is directly connected, Loopback0
         172.16.0.0/24 is subnetted, 6 subnets
    O       172.16.252.0 [110/128] via 172.16.250.2, 00:07:46, Serial0
    C       172.16.250.0 is directly connected, Serial0
    C       172.16.251.0 is directly connected, Serial1
    O       172.16.50.0 [110/74] via 172.16.250.2, 00:07:46, Serial0
    C       172.16.1.0 is directly connected, Ethernet0
    O       172.16.100.0 [110/192] via 172.16.250.2, 00:07:46, Serial0
```

Note that the routing table shows that _NewYork_ learns `10.0.3.0` as an external route whereas `10.0.1.0` and `10.0.2.0` are learned as inter-area routes (lines 17-19) -- this is because inter-area routes are preferred over external routes. The OSPF order of route preference, from most preferred to least preferred, is as follows: intra-area, inter-area, type 1 external, type 2 external.

#### Router LSA (type 1)

A router LSA describes the advertising router’s directly connected links. Routers _Chicago_, _Ames_, _NewYork_, and _NewYork2_ advertise router LSAs that are flooded throughout area 0. _NewYork_’s LS database holds router LSAs from all these routers, but for the sake of brevity I’ll show only the contents of the LSA from _NewYork2_.

The number of links (as in line 20 in the upcoming code block) described in the LSA is 1. Although _NewYork2_ has two directly connected links -- an Ethernet segment and a serial link -- only the Ethernet segment is described in the LSA to _NewYork_. This is because the serial link is in area 1 and router LSAs do not cross OSPF area boundaries.

The link described is a _transit network_ (line 21), implying that there are multiple routers on the link. Other link types are point-to-point (for serial links), stub network (for a network with only one router), and virtual link (for OSPF virtual links).

The value of the link ID field depends on the type of link being described, as shown in [Table 6-4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-TABLE-4 "Table 6-4. Link type and link ID").

Table 6-4. Link type and link ID

|Link type|Link ID|
|---|---|
|Point-to-point|Neighbor’s router ID|
|Transit network|DR’s IP address on network|
|Stub network|IP network number or subnet number|
|Virtual link|Neighbor’s router ID|

In our example, the DR is _NewYork_, so the link ID (in line 22) contains _NewYork_’s IP address.

The contents of the link data field also depend on the link type, as shown in [Table 6-5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-TABLE-5 "Table 6-5. Link type and link data").

Table 6-5. Link type and link data

|Link type|Link data|
|---|---|
|Point-to-point|IP address of network interface|
|Transit network|IP address of network interface|
|Stub network|IP network number or subnet number|
|Virtual link|MIB II ifIndex for the router’s interface|

In our example, the link data field (in line 23) specifies the IP address of _NewYork2_:

```
   NewYork#sh ip ospf database router

          OSPF Router with ID (192.168.1.1) (Process ID 10)

   Routing Bit Set on this LSA

     LS age: 209

     Options: (No TOS-capability, DC)
     LS Type: Router Links
     Link State ID: 192.168.1.4

     Advertising Router: 192.168.1.4

     LS Seq Number: 800000FF
     Checksum: 0x2BA1
     Length: 36
     Area Border Router
     AS Boundary Router
20    **`Number of Links: 1`**     

21     **`Link connected to: a Transit Network`** 
22      **`(Link ID) Designated Router address: 172.16.1.1`** 
23      **`(Link Data) Router Interface address: 172.16.1.2`** 
         Number of TOS metrics: 0
          TOS 0 Metrics: 10
```
#### Network LSA (type 2)

A network LSA describes broadcast/NBMA networks. The network LSA is originated by the DR and describes all attached routers.

The LSA in the following example is self-originated, as seen in the advertising router field (line 24), which shows _NewYork_’s own router ID. The network LSA describes the mask on the multi-access network (line 25) and the IP addresses of the routers on the multi-access network (lines 26 and 27).

```
   NewYork#sh ip ospf database network

          OSPF Router with ID (192.168.1.1) (Process ID 10)


              Net Link States (Area 0)

     Routing Bit Set on this LSA
     LS age: 1728
     Options: (No TOS-capability, DC)
     LS Type: Network Links
     Link State ID: 172.16.1.1 (address of Designated Router)            
24   **`Advertising Router: 192.168.1.1`**  
     LS Seq Number: 800000F4
     Checksum: 0x172B
     Length: 32
25   **`Network Mask: /24`** 
26    **`Attached Router: 192.168.1.1`**  
27    **`Attached Router: 192.168.1.4`**
```
#### Summary LSA (type 3)

A summary LSA is advertised by an ABR and describes inter-area routes.

The summary LSAs in the following example are originated by _NewYork2_ (`192.168.1.4`) and describe routes to `10.0.1.0` and `10.0.2.0`, respectively. The link state ID describes the summary network number (lines 28 and 31). Note that each LSA describes just one summary network number.

```
   NewYork#sh ip ospf database summary

          OSPF Router with ID (192.168.1.1) (Process ID 10)


               Summary Net Link States (Area 0)

     Routing Bit Set on this LSA
     LS age: 214
     Options: (No TOS-capability, DC)
     LS Type: Summary Links(Network)
28   **`Link State ID: 10.0.1.0 (summary Network Number)`**
29   **`Advertising Router: 192.168.1.4`**  
     LS Seq Number: 80000062
     Checksum: 0x85A
     Length: 28
30   **`Network Mask: /24`**                                
       TOS: 0     Metric: 1785 

     Routing Bit Set on this LSA
     LS age: 214
     Options: (No TOS-capability, DC)
     LS Type: Summary Links(Network)
31   **`Link State ID: 10.0.2.0 (summary Network Number)`**
32   **`Advertising Router: 192.168.1.4`**        
     LS Seq Number: 80000061
     Checksum: 0x62F5
     Length: 28
33   **`Network Mask: /24`**                                
    TOS: 0     Metric: 1795
```
#### ASBR summary LSA (type 4)

An ASBR summary LSA describes the route to the ASBR. The mask associated with a type 4 LSA is 32 bits long because the route advertised is to a host -- the host being the ASBR. ASBR summary LSAs are originated by ABRs.

The link state ID (line 34) in this example describes the router ID of _Paris_, which is the ASBR redistributing RIP into OSPF. The advertising router is the ABR -- _NewYork2_ (line 35).

```
   NewYork#sh ip ospf database asbr-summary

          OSPF Router with ID (192.168.1.1) (Process ID 10)


              Summary ASB Link States (Area 0)

     Routing Bit Set on this LSA
     LS age: 115
     Options: (No TOS-capability, DC)
     LS Type: Summary Links(AS Boundary Router)
34   **`Link State ID: 192.168.1.5 (AS Boundary Router address)`**
35   **`Advertising Router: 192.168.1.4`**      
     LS Seq Number: 80000061
     Checksum: 0x9A63
     Length: 28
     Network Mask: /0
       TOS: 0     Metric: 1785
```
#### External LSA (type 5)

External LSAs originate at ASBRs and describe routes external to the OSPF process. External LSAs are flooded throughout the OSPF network, with the exception of stub areas.

Network `10.0.1.0` is learned via RIP from _NewYork2_, which floods an external LSA with a link state ID of `10.0.1.0`. Interestingly, `10.0.1.0` is also known as an inter-area route (see the section [Section 6.4.5.3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-SECT-4.5.3 "Summary LSA (type 3)")). Router _NewYork_ prefers the IA route (see line 19) but will keep the external LSA in its topological database. The advertising router (line 37) is _Paris_, the ASBR, which redistributes RIP into OSPF. The forwarding address (in line 39) is `0.0.0.0`, indicating that the destination for `10.0.1.0` is the ASBR. The LSA (in line 40) specifies an external route tag of 0, which indicates a type 1 external route; a value of 1 would indicate a type 2 external route.

```
   NewYork#sh ip ospf database external

          OSPF Router with ID (192.168.1.1) (Process ID 10)


               Type-5 AS External Link States

     LS age: 875
     Options: (No TOS-capability, No DC)
     LS Type: AS External Link
36   **`Link State ID: 10.0.1.0 (External Network Number )`**
37   **`Advertising Router: 192.168.1.5`**         
     LS Seq Number: 80000060
     Checksum: 0x6F27
     Length: 36
38   **`Network Mask: /24`**                                
      Metric Type: 2 (Larger than any link state path)
      TOS: 0 
      Metric: 100 
39    **`Forward Address: 0.0.0.0`**
40    **`External Route Tag: 0`**             
   ...
```

Note that _NewYork_’s external database contains two other LSAs -- with link state IDs of `10.0.2.0` and `10.0.3.0` -- which were not shown here.

#### NSSA external LSA (type 7)

NSSA external LSAs describe routes external to the OSPF process. However, unlike type 5 external LSAs, NSSA external LSAs are flooded only within the NSSA.

There are no type 7 LSAs in this network. In fact, there aren’t even any NSSAs in this network:

```
NewYork#sh ip  ospf database nssa-external

       OSPF Router with ID (192.168.1.1) (Process ID 10)
```

The format of the NSSA external LSA is identical to that of the AS external LSA, except for the forwarding address field. The forwarding address field in an NSSA external LSA always indicates the address to which traffic should be forwarded.

#### Flooding of LSAs

LSAs are generated every 30 minutes, or sooner if there is a change in the state of a link. LSAs are exchanged between routers that have established _adjacency_, as was described earlier.

The rules for the flooding of LSAs are governed by the hierarchical structure of OSPF, as given in [Table 6-6](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-TABLE-6 "Table 6-6. Rules for the flooding of LSAs").

Table 6-6. Rules for the flooding of LSAs

|LSA type|Originating router|Area in which flooded|
|---|---|---|
|Router LSA (type 1)|Every router|Router’s local area.|
|Network LSA (type 2)|DR|Router’s local area.|
|Summary LSA (type 3)|ABR|Nonlocal area.|
|ASBR summary LSA (type 4)|ASBR|All areas except stub area, totally stubby area, or NSSA.|
|External LSA (type 5)|ASBR|All areas except stub area, totally stubby area, or NSSA.|
|NSSA external LSA (type 7)|ASBR|Router’s local area. NSSA external LSA may be forwarded by ABR as a type 5 LSA.|

## Route Summarization

RIP-1 and IGRP automatically summarize subnets into a major network number when crossing a network-number boundary. OSPF does not automatically summarize routes. Route summarization in OSPF must be manually configured on an ABR or an ASBR. Further, OSPF allows route summarization on any bit boundary (unlike RIP and IGRP, which summarize only classful network numbers).

Summarizing routes keeps the routing tables smaller and easier to troubleshoot. However, route summarization in OSPF is not just a nice thing to do -- it is necessary to reduce the size of the OSPF topology database, especially in a large network. A large topology database requires a large amount of router memory, which slows down all processes, including SPF calculations.

To allow summarization at ABRs and ASBRs, IP addresses must be carefully assigned. First, allocate enough addresses to each area to allow for expansion. Then set a bit boundary on which to summarize routes. This is easier said than done. Most network engineers inherit a network with a haphazard mess of addresses and changing requirements.

### Summarizing at the ABR (Inter-Area Summarization)

Consider TraderMary’s network in [Figure 6-10](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-FIG-10 "Figure 6-10. TraderMary’s network with a French extension"). Network `10.0.0.0` exists in area 1, and network `172.16.0.0` exists in area 0. Let’s see how we can summarize on these area boundaries.

The command to summarize at an ABR is:

```
area _`area-id`_ range _`address mask`_
```

where _area-id_ is the area whose routes are to be summarized, _address_ is a network number, and _mask_ specifies the number of bits in _address_ to summarize.

The OSPF configuration on _NewYork2_ can now be modified to summarize `172.16.0.0` routes into area 1 (line 41) and `10.0.0.0` routes into area (line 42).

```
   hostname NewYork2
   ...
   router ospf 10
    redistribute static metric 10
    network 172.16.0.0 0.0.255.255 area 0
    network 10.0.0.0 0.255.255.255 area 1
41  **`area 0 range 172.16.0.0 255.255.0.0`**                 
42  **`area 1 range 10.0.0.0 255.0.0.0`**
```

The routing table in _Paris_ is now as follows. Note that _Paris_ has only one summary route for `172.16.0.0/16` (line 43).

```
    Paris#show ip route
    ...
    10.0.0.0/24 is subnetted, 2 subnets
    C       10.0.2.0 is directly connected, Ethernet0
    C       10.0.1.0 is directly connected, Serial1
         192.168.1.0/32 is subnetted, 1 subnets
    C       192.168.1.5 is directly connected, Loopback0
43  **`O IA 172.16.0.0/16 [110/74] via 10.0.1.2, 1d23h, Serial1`**
```

The routing table for _NewYork_ is now as follows. Note that _NewYork_ has only one summarized route for `10.0.0.0/8` (line 44).

```
    NewYork#sh ip route
    ...
    O IA 10.0.0.0/8 [110/1795] via 172.16.1.2, 1d23h, Ethernet0
         192.168.1.0/32 is subnetted, 1 subnets
    C       192.168.1.1 is directly connected, Loopback0
         172.16.0.0/24 is subnetted, 6 subnets
    O       172.16.252.0 [110/128] via 172.16.250.2, 1d23h, Serial0
    C       172.16.250.0 is directly connected, Serial0
    C       172.16.251.0 is directly connected, Serial1
    O       172.16.50.0 [110/74] via 172.16.250.2, 1d23h, Serial0
    C       172.16.1.0 is directly connected, Ethernet0
44  **`O       172.16.100.0 [110/192] via 172.16.250.2, 1d23h, Serial0`**
```

When an EIGRP router summarizes, it automatically builds a route to _null0_ for the summarized route. (This is explained in detail in the section [Section 4.5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s05.html "Route Summarization") in [Chapter 4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04.html "Chapter 4. Enhanced Interior Gateway Routing Protocol (EIGRP)")). The router to _null0_ prevents packets that do not match a specific entry in the routing table from following a default route. (The route to _null0_ causes the packet to be dropped). However, as you saw earlier, OSPF does not build a null route. You may want to manually add a static route to _null0_ on the ABR.

### Summarizing at the ASBR (or External Route Summarization)

In the configuration in [Figure 6-10](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-FIG-10 "Figure 6-10. TraderMary’s network with a French extension"), _Paris_ is the ASBR redistributing RIP into OSPF. Note from the figure that the RIP network contains routes in the network `10.3.0.0/24` (the RIP subnets may be `10.3.1.0/24`, `10.3.2.0/24`, `10.3.3.0/24`, ... `10.3.255.0/24`). It is desirable to summarize `10.3.0.0/16` into the OSPF network rather than carrying the individual subnets.

The routes being redistributed into OSPF can be summarized at the ASBR (which is _Paris_ in the previous example) using the following command:

```
summary-address _`address mask`_
```

where _address_ defines a summary IP address and _mask_ describes the range of addresses.

Router _Paris_ may thus be configured as follows to summarize `10.3.0.0/16` into the OSPF network:

```
hostname Paris
!
interface Loopback0
 ip address 192.168.1.5 255.255.255.255
!
interface Ethernet0
 ip address 10.0.2.1 255.255.255.0
!
interface Serial1
 ip address 10.0.1.1 255.255.255.0
!
router ospf 10
 **`summary-address 10.3.0.0 255.255.252.0`**
 redistribute rip metric 100 subnets
 network 10.0.0.0 0.255.255.255 area 1
!
router rip
 network 10.0.0.0
```

The LS database will now contain a single external LSA with a link state ID of `10.3.0.0` advertised by _Paris_.

## Default Routes

Earlier chapters showed how a default route could be used for branch office connectivity. A default route can also be used when connecting to the Internet to represent _all_ the routes in the Internet. Let’s say that TraderMary established a connection from _NewYork2_, _Serial2_ (line 45) to an Internet service provider (ISP). A static default route is also installed on _NewYork2_ (line 47), pointing to the ISP.

_NewYork2_ is configured as in line 46 to source a default route. The keyword _always_ implies that the default route must be originated whether or not the default route is up. _metric-value_ is the metric to associate with the default route (the default for this field is 10). Note that this redistribution of a default route into OSPF makes _NewYork2_ an ASBR. The keyword _metric-type_ can be set to 1 or 2 to specify whether the default route is external type 1 or 2 (the default is 2).

```
    hostname NewYork2
    !
45  **`interface Serial2`**      
    description Connection to the ISP
    ip address 146.146.1.1 255.255.255.0
    !
    router ospf 10
    network 172.16.0.0 0.0.255.255 area 0
46  **`default-information originate always metric-value 20 metric-type 1`**
    !
47  **`ip route 0.0.0.0 0.0.0.0 interface serial2`**
```

Since the keyword _always_ was specified, the default route will not disappear from the OSPF routing table if _Serial2_ (the link to the ISP) is down. If TraderMary has two (or more) routers connecting to ISPs and each router announced a default route into OSPF, do not use the _always_ keyword -- if one ISP connection is lost, traffic will find its way to the other ISP connection.

To ensure that the default route is always announced (even if _Serial2_ goes down) choose the _always_ option.

A default route of type 1 includes the internal cost of reaching the ASBR. If TraderMary has multiple Internet connections, announcing a default route from each with a metric type of 1 would have the advantage that any router in the network would find the closest ASBR.

## Virtual Links

TraderMary is planning to establish a new office in Paris with an area ID of 2. The first router in area 2 will be called _Paris2_. A direct circuit needs to be established from _NewYork2_ (the ABR) to _Paris2_, since all OSPF areas must connect directly to the backbone (area 0). This international circuit has a long installation time. And, since a physical path is already available to area 2 via area 1, you may ask if OSPF provides some mechanism to activate area 2 before the _NewYork2_ → _Paris2_ circuit can be installed. The answer is yes. OSPF defines virtual links (VLs) which can extend the backbone area. Area 2 will directly attach to the backbone via the VL. A VL may be viewed as a point-to-point link belonging to area 0. The endpoints of a VL must be ABRs.

In our example in [Figure 6-11](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s07.html#iprouting-CHP-6-FIG-11 "Figure 6-11. Virtual link to area 2"), a virtual link may be defined from _NewYork2_ to _Paris2_ through area 1.

![[Pasted image 20241110184415.png]]
Figure 6-11. Virtual link to area 2

The syntax for configuring a virtual link is as follows:

```
area _`area-id`_ virtual-link _`router-id`_ [hello-interval _`seconds`_] [retransmit-interval  
_`seconds`_] [transmit-delay _`seconds`_] [dead-interval _`seconds`_] [[authentication-key _`key`_] | 
[message-digest-key _`keyid`_ md5 _`key`_]]
```

where _area-id_ specifies the transit area and _router-id_ specifies the ABR with which the neighbor relationship is to be established. The four timers refer to the time between hello packets (default is 10 s), the time between LSA retransmissions (default is 5 s), the time by which LSAs are aged when they transmit this interface (default is 1 s), and the router dead-interval (default is four times the hello-interval). The parameter _key_ is a string of characters up to 8 bytes long, _keyid_ is in the range 1-255, and _key_ is an alphanumeric string up to 16 characters in length.

Remember that a virtual link can be created only between ABRs and can traverse only one area. _Paris2_ is an ABR because it has connectivity to areas 1 and 2. _NewYork2_ is an ABR with connectivity to areas and 1. Thus, a virtual link may be configured between _Paris2_ and _NewYork2_ traversing area 1:

```
hostname Paris2
!
interface Loopback1
 ip address 192.168.1.6 255.255.255.255 
!
interface Loopback2
 ip address 192.168.2.1 255.255.255.0
!
interface Ethernet0
 ip address 10.0.2.2 255.255.255.0
!
router ospf 10
 network 10.0.0.0 0.255.255.255 area 1
 network 192.168.2.0 0.0.0.255 area 2
 **`area 1 virtual-link 192.168.1.4`**

hostname NewYork2
!
interface Loopback0
 ip address 192.168.1.4 255.255.255.255
!
interface Ethernet0
 ip address 172.16.1.2 255.255.255.0
!
interface Serial1
 ip address 10.0.1.2 255.255.255.0
 bandwidth 56
!
router ospf 10
 redistribute static metric 10
 network 172.16.0.0 0.0.255.255 area 0
 network 10.0.0.0 0.255.255.255 area 1
 **`area 1 virtual-link 192.168.1.6`**
```

The status of the virtual link can be verified as follows:

```
Paris2#sh ip ospf virtual-link
**`Virtual Link to router 192.168.1.4 is up`**
  Transit area 1, via interface Ethernet0, Cost of using 74
  Transmit Delay is 1 sec, State POINT_TO_POINT,
  Timer intervals configured, Hello 10, Dead 40, Wait 40, Retransmit 5
    Hello due in 0:00:00
    **`Adjacency State FULL`**

NewYork2#show ip ospf virtual-link
**`Virtual Link OSPF_VL0 to router 192.168.1.6 is up`**
  Run as demand circuit
  DoNotAge LSA not allowed (Number of DCbitless LSA is 8).
  Transit area 1, via interface Serial1, Cost of using 1795
  Transmit Delay is 1 sec, State POINT_TO_POINT,
  Timer intervals configured, Hello 10, Dead 40, Wait 40, Retransmit 5
    Hello due in 00:00:05
    **`Adjacency State FULL`**
```

VLs cannot traverse stub areas (or totally stubby areas or NSSAs). This is because VLs belong to area 0, and in order for area to route correctly it must have the complete topology database. Stub areas do not contain the complete topology database.

VLs find one other use in OSPF -- they may be used to repair the network in the event that an area loses its link to the backbone. For example, in [Figure 6-4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s03.html#iprouting-CHP-6-FIG-4 "Figure 6-4. OSPF architecture: a high-level view"), the loss of the link _R1_ → _R4_ will isolate area 2 from the rest of the network. Until the _R1_ → _R4_ link is repaired, a VL may be defined between _R4_ and _R5_ to join area 2 to the backbone.

## Demand Circuits

The cost of a demand circuit, such as an ISDN link or a dial-up line, is dependent on its usage. It is desirable to use a demand circuit only for user traffic and not for overhead such as OSPF hellos or periodic LSAs. RFC 1793 describes modifications to OSPF that allow the support of demand circuits. This is an optional capability in OSPF; a router will set the DC bit in the options field if it supports the capability. Routers that support the capability will also set the high bit of the LS age field to 1 to indicate that the LSA should not be aged. This bit is also referred to as the do-not-age bit. OSPF demand circuits suppress periodic hellos and LSAs, but a topology change will still activate the demand circuit since LSA updates are required to keep the LS database accurate. Since any large network is likely to experience frequent topology changes, it may be prudent to define demand circuits in stub areas. Stub areas have a limited topology database and hence are shielded from frequent topology changes.

If a demand circuit is created in a stub area, all routers in the stub area must support the DC option -- routers that do not support demand circuits will misinterpret the age field (as the high bit is set). An LSA with the DC bit set to 1 is flooded into an area only if all LSAs in the database have their DC bits set to 1.

To configure an interface as a demand circuit, enter the following command in interface configuration mode on one end of the demand circuit:

```
ip ospf demand-circuit
```

LSA updates will bring up the demand circuit only if there is a change in topology.

## Stub, Totally Stubby, and Not So Stubby Areas

External LSAs are flooded through the OSPF backbone as well as through all regular areas. Let’s test this using TraderMary’s network of [Figure 6-10](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html#iprouting-CHP-6-FIG-10 "Figure 6-10. TraderMary’s network with a French extension"). A static route for `192.168.3.0` is defined (pointing to _null0_) on _Chicago_ and redistributed into OSPF. Router _Chicago_ then advertises an external LSA with a link state ID of `192.168.3.0`:

```
hostname Chicago
!
router ospf 10
 **`redistribute static metric 100 metric-type 1 subnets`**
 network 172.16.0.0 0.0.255.255 area 0
!
**`ip route 192.168.3.0 255.255.255.0 Null0`**
```

The LSA is flooded to all routers in the network. Let’s check _Paris_ as an instance:

```
Paris#sh ip ospf database external

       OSPF Router with ID (192.168.1.5) (Process ID 10)


           AS External Link States

  Routing Bit Set on this LSA
  LS age: 158
  Options: (No TOS-capability)
  LS Type: AS External Link
  **`Link State ID: 192.168.3.0 (External Network Number )`**
  Advertising Router: 192.168.1.3
  LS Seq Number: 80000001
  Checksum: 0x8F67
  Length: 36
  Network Mask: /24
    Metric Type: 1 (Comparable directly to link state metric)
    TOS: 0 
    Metric: 100 
    Forward Address: 0.0.0.0
    External Route Tag: 0
```

The route to `192.168.3.0` also appears in the routing table:

```
Paris#sh ip route
...
Gateway of last resort is not set
...
**`O E1 192.168.3.0/24 [110/302] via 10.0.1.2, 00:02:08, Serial1`**
...
```

Flooding external LSAs throughout an OSPF network may be a waste of resources. Stub areas block the flooding of external LSAs, as we will see in the next section.

### Stub Areas

Referring to [Figure 6-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06.html#iprouting-CHP-6-FIG-1 "Figure 6-1. Overview of OSPF areas"), the router in area 1 that connects to the RIP network floods external LSAs into the network. It appears that nothing is gained by importing external LSAs into areas 2 and 3, which can point all external routes to their ABRs using default routes. Representing every external LSA in areas 2 and 3 would be a waste of resources. With this in mind, OSPF defines _stub areas_ . When an area is defined as a stub area, all external LSAs are blocked at the ABRs, and, in place, the ABRs source a single default route into the stub area.

All routers in a stub area must be configured as stub routers. Stub routers form adjacencies only with other stub routers and do not propagate external LSAs. (How does a router know if its neighbor is a stub router? The E bit in the hello packet is turned to zero if the router is a stub router).

Area 1 in TraderMary’s network can be made stubby via the following configuration changes:

```
hostname NewYork2
...
router ospf 10
network 172.16.0.0 0.0.255.255 area 0
network 10.0.0.0 0.255.255.255 area 1
**`area 1 stub`**

hostname Paris
...
router ospf 10
 redistribute rip
 network 10.0.0.0 0.255.255.255 area 1
 **`area 1 stub`**
```

The routing table for _Paris_ now shows a default route pointing to the ABR (_NewYork2_) but does not show the external route to 192.168.3.0 (sourced by _Chicago_):

```
Paris#sh ip route
...
Gateway of last resort is 10.0.1.2 to network 0.0.0.0
...
O*IA 0.0.0.0/0 [110/65] via 10.0.1.2, 00:00:35, Serial1
O IA 172.16.0.0/16 [110/74] via 10.0.1.2, 1d23h, Serial1
...
```

After making this change, however, we will find that the network has lost connectivity to `10.0.3.0`, which represents the RIP external network connecting to router _Paris_. The reason for this is rather obvious: stub areas do not propagate external LSAs. In other words, an ASBR cannot belong to a stub area.

The other major restriction with stub areas is that they cannot support virtual links, because they don’t have the complete routing table. An area that needs to support a VL cannot be a stub area.

Any area that does not contain an ASBR (i.e., does not support a connection to an external network) and is not a candidate for supporting a virtual link should be made a stub area.

There is one major disadvantage to configuring an area as a stub area. When multiple ABRs source a default route, the routers in the stub area may fail to recognize the shortest path to the destination network. This may help determine whether you choose to implement an area as a regular area or as a stub area.

### Totally Stubby Areas

Totally stubby areas carry the concept of stub areas further by blocking all summary LSAs in addition to external LSAs.

In the configuration in the previous section, where _Paris_ is configured as a stub area, the LS database for _Paris_ will not show external LSAs but will still show all summary LSAs, so _Paris_’s routing table still shows the summarized inter-area route to `172.16.0.0/16`. If _NewYork2_ did not summarize the `172.16.0.0` subnets, _Paris_ would show all six `172.16.0.0` subnets: `172.16.1.0/24`, `172.16.50.0/24`, `172.16.100.0/24`, `172.16.250.0/24`, `172.16.251.0/24`, and `172.16.252.0/24`. Totally stubby areas, unlike stub areas, replace all inter-area routes (in addition to external routes) with a default route.

Area 1 can be configured as a totally stubby area by modifying the configuration of _NewYork2_ as follows. No change is required to router _Paris_.

```
hostname NewYork2
!
router ospf 10
 redistribute static metric 10
 network 172.16.0.0 0.0.255.255 area 0
 network 10.0.0.0 0.255.255.255 area 1
 **`area 1 stub no-summary`**
```

_Paris_’s routing table now does not contain any IA routes (other than the default sourced by _NewYork2_):

```
Paris#sh ip route
...
Gateway of last resort is 10.0.1.2 to network 0.0.0.0

     10.0.0.0/24 is subnetted, 2 subnets
C       10.0.2.0 is directly connected, Ethernet0
C       10.0.1.0 is directly connected, Serial1
     192.168.1.0/32 is subnetted, 1 subnets
C       192.168.1.5 is directly connected, Loopback0
O*IA 0.0.0.0/0 [110/65] via 10.0.1.2, 00:00:23, Serial1
```

Totally stubby areas have the same restrictions as stub areas -- no ASBRs (no external LSAs) and no virtual links. Also, like stub areas, totally stubby areas see all ABRs as equidistant to all destinations that match the default route. When multiple ABRs source a default route, the routers in the totally stubby area may not recognize the shortest path to the destination network.

### NSSAs

What if a stub area needs to learn routes from another routing protocol? For example, _Paris_ -- in area 1 -- may need to learn some RIP routes from a legacy network. NSSAs -- as specified in RFC 1587 -- allow external routes to be imported into an area without losing the character of a stub area (i.e., without importing any external routes from the backbone area).

NSSAs import external routes through an ASBR in type 7 LSAs. Type 7 LSAs are flooded within the NSSA. Type 7 LSAs may optionally be flooded into the entire OSPF domain as a type 5 LSAs by the ABR(s) or be blocked at the ABR(s). As with any stub area, NSSAs do not import type 5 LSAs from the ABR.

The option (of whether or not to translate a type 7 LSA into a type 5 LSA at the NSSA ABR) is indicated in the P bit (in the options field) of the type 7 LSA. If this bit is set to 1, the LSA is translated by the ABR into a type 5 LSA to be flooded throughout the OSPF domain. If this bit is set to 0, the LSA is not advertised outside the NSSA area.

All routers in the NSSA must be configured with the _nssa_ keyword (line 48):

```
    hostname NewYork2
    !
    router ospf 10
     redistribute static metric 10
     network 172.16.0.0 0.0.255.255 area 0
     network 10.0.0.0 0.255.255.255 area 1
48   **`area 1 nssa`**
```

There are three optional keywords for NSSA configuration:

```
   area 1 nssa ?
49   **`default-information-originate`** 
50   **`no-redistribution`**                                
51   **`no-summary`**
```

When configured on the NSSA ABR, the _default-information-originate_ keyword (line 49) causes the ABR to source a default route into the NSSA.

The _no-redistribution_ keyword (line 50) is useful on NSSA ABRs that are also ASBRs. The _no-redistribution_ keyword stops the redistribution of external LSAs (from the other AS) into the NSSA.

The _no-summary_ keyword (line 51) gives you another oxymoron -- it makes the NSSA a totally stubby NSSA, so no type 3 or 4 LSAs are sent into the area.

NSSAs are thus a variant of stub areas with one less restriction -- external connections are allowed. In all other respects, NSSAs are just stub areas.

## NBMA Networks

Remember how a DR is elected -- basic to DR election is the broadcast or multicast capability of the underlying network. NBMA networks such as Frame Relay or X.25 have no inherent broadcast or multicast capability, but they can simulate a broadcast network if fully meshed. However, a fully meshed network with _n_ nodes requires _n_ x (_n_-1)/2 virtual circuits. The cost of _n_ x (_n_-1)/2 virtual circuits may be unpalatable, and besides, the failure of a single virtual circuit would disrupt this full mesh.

One option around a fully meshed network is to (statically) configure the DR for the network. The DR will then advertise the NBMA network as a multi-access network using a single IP subnet in a network LSA.

Another option is to configure the network as a set of point-to-point networks. This is simpler to configure, manage, and understand. However, each point-to-point network wastes an IP subnet. So what? You can use VLSM in OSPF, with a two-bit subnet for each point-to-point network. That is a good argument. However, the trade-off is the processing overhead of an LSA for each point-to-point network.

Let’s look at examples of each of these options.

_NewYork2_ is set up with a serial interface to support Frame Relay PVCs to offices in Miami and New Orleans, as shown in [Figure 6-12](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s10.html#iprouting-CHP-6-FIG-12 "Figure 6-12. TraderMary’s Frame Relay network").

![[Pasted image 20241110184849.png]]
Figure 6-12. TraderMary’s Frame Relay network

The command **ip ospf network broadcast** (lines 52, 53, and 55) makes OSPF believe that the attached network is multi-access, like an Ethernet segment. However, since the network has no true broadcast capability, the priorities on _NewYork2_, _Miami_, and _NewOrleans_ must be specified to force _NewYork2_ to be the DR on the NBMA network. _NewYork2_ will become the DR while the state of the interface on _Miami_ and _NewOrleans_ will be DRother (implying that the interface has not been elected the DR). _NewYork2_ uses the default priority of 1. _Miami_ and _NewOrleans_ are configured with a priority value of (lines 54 and 56), which makes them ineligible for DR election.

```
   hostname NewYork2
   !
   interface Serial3
    ip address 192.168.10.2 255.255.255.0     
    encapsulation frame-relay
52  **`ip ospf network broadcast`**                 
    ip ospf hello-interval 30
    keepalive 15
    frame-relay lmi-type ansi
   !
   router ospf 10
   network 192.168.10.0 0.0.0.255 area 0


   hostname Miami
   !
   interface Serial0
    no ip address
    encapsulation frame-relay
    keepalive 15
    frame-relay lmi-type ansi
   !
   interface Serial0.1 point-to-point
    ip address 192.168.10.3 255.255.255.0
53  **`ip ospf network broadcast`** 
    ip ospf hello-interval 30
54  **`ip ospf priority 0`**                             
    frame-relay interface-dlci 100   
   !
   router ospf 10
    network 192.168.10.0 0.0.0.255 area 0


   hostname NewOrleans
   !
   interface Serial0
    no ip address
    encapsulation frame-relay
    bandwidth 1544
    keepalive 15
    lat enabled
    frame-relay lmi-type ansi
   !
   interface Serial0.1 point-to-point
    ip address 192.168.10.1 255.255.255.0
55  **`ip ospf network broadcast`**                          
    ip ospf hello-interval 30
56  **`ip ospf priority 0`**                        
    frame-relay interface-dlci 200   
   !
   router ospf 10
   network 192.168.10.0 0.0.0.255 area 0
```

IOS releases prior to 10.0 did not support the command **ip ospf network broadcast** and required the static configuration of neighbors and their priorities:

```
neighbor _`ip-address`_ [priority _`number`_] [poll-interval _`seconds`_]
```

where _ip-address_ is the IP address of the neighbor, _number_ is the neighbor’s priority (0-255), and _seconds_ is the dead router poll interval.

The NBMA network may be modeled as a collection of point-to-point networks. Configure the routers the same way, but configure the interfaces as point-to-multipoint instead of broadcast and do not specify the OSPF priority, since a point-to-multipoint network does not elect a DR (the hello protocol is used to find neighbors):

```
ip ospf network point-to-multipoint
```

The point-to-multipoint network consumes only one IP subnet but creates multiple host routes.

You can also use subinterfaces to model the NBMA network as a collection of point-to-point networks. Routers at the ends of a point-to-point subinterface always form adjacency, much like routers at the ends of a serial interface. No DR election takes place. Since OSPF supports VLSM, one cannot argue that this will waste IP address space. However, using point-to-point subinterfaces in lieu of a single broadcast network generates LSAs for every subinterface, which adds to the processing overhead.

## OSPF Design Heuristics

The following sections provide a partial and ad hoc checklist to use when executing an OSPF design. As with any other discipline, the engineer will do best if he spends time understanding the details of OSPF and then designs his network as simply as possible.

### OSPF Hierarchy

Building a large, unstructured OSPF network is courting disaster. The design of the OSPF network must be clearly defined: all changes in the OSPF environment must bear the imprint of the OSPF architecture. For example, when adding a new router, the network engineer must answer the following questions:

- Will the router be an area router, a stub router, or an ABR?
    
- If the router is an ABR or an ASBR, what routes should the router summarize?
    
- What impact would the failure of the router have on OSPF routing?
    
- Will this router be a DR/BDR?
    
- How will this router affect the performance of other OSPF routers?
    

### IP Addressing

IP addresses must be allocated in blocks that allow route summarization at ABRs. The address blocks must take into account the number of users in the area, leaving room for growth. VLSM should be considered when planning IP address allocation.

### Router ID

Use loopback addresses to assign router IDs. Choose the router IDs carefully -- the router ID will impact DR/BDR election on all attached multi-access networks. Keep handy a list of router IDs and router names. This will make it easier to troubleshoot the network.

### DR/BDR

Routers with low processor/memory/bandwidth resources should be made DR-ineligible. A router that becomes the DR/BDR on multiple networks may see high memory/CPU utilization.

### Backbone Area

Since all inter-area traffic will traverse the backbone, ensure that there is adequate bandwidth on the backbone links. The backbone area will typically be composed of the highest-bandwidth links in the network, with multiple paths between routers.

The backbone should have multiple paths between any pair of nonbackbone areas. A partitioned backbone will disrupt inter-area traffic -- ensure that there is adequate redundancy in the backbone.

Use the backbone solely for inter-area traffic -- do not place users or servers on the backbone.

### Number of Routers in an Area

The maximum number of routers in an area depends on a number of factors -- number of networks, router CPU, router memory, etc. -- but Cisco documentation suggests that between 40 and 50 is a reasonable number. However, it is not uncommon to have a couple of hundred routers in an area, although problems such as flaky links may overload the CPU of the routers in the area. As a corollary of the previous argument, if you think that the total number of routers in your network will not exceed 50, all the routers can be in area 0.

### Number of Neighbors

If the number of routers on a multi-access network exceeds 12 to 15 and the DR/BDR is having performance problems, look into a higher-horsepower router for the DR/BDR. Note that having up to 50 routers on a broadcast network is not uncommon. The total number of neighbors on all networks should not exceed 50 or so.

### Route Summarization

To summarize the routes:

- Allocate address blocks for each area based on bit boundaries. As areas grow, keep in mind that the area may ultimately need to be split into two. If possible, allocate addresses within an area in contiguous blocks to allow summarization at the time of the split.
    
- Summarize into the backbone at the ABR (as opposed to summarizing into the nonbackbone area). This reduces the sizes of the LS database in the backbone area and the LS databases in the nonbackbone areas.
    
- Route summarization has the advantage that a route-flap in a subnet (that has been summarized) does not trigger an LSA to be flooded, reducing the OSPF processing overhead.
    
- If an area has multiple ABRs and one ABR announces more specific routes, all the traffic will flow to that router. This is good if this is the desired effect. Otherwise, if you intend to use all ABRs equally, all ABRs must have identical summary statements.
    
- Summarize external routes at the ASBR.
    
- Golden rule: summarize, summarize, summarize.
    

### VLSM

OSPF LSA records carry subnet masks; the use of VLSM is encouraged to conserve the available IP address space.

### Stub Areas

An area with only one ABR is an ideal candidate for a stub area. Changing the area into a stub area will reduce the size of the LS database without the loss of any useful routing information. Remember that stub areas cannot support VLs or type 5 LSAs.

### Virtual Links

Design the network so that virtual links are not required. VLs should be used only as emergency fixes, not as a part of the design.

### OSPF Timers

In an all-Cisco network environment, the OSPF timers (hello-interval, dead-interval, etc.) can be left to their default values; in a multivendor environment, however, the network engineer may need to adjust the timers to make sure they match.

## Troubleshooting OSPF

OSPF is a complex organism and hence can be difficult to troubleshoot. However, since the operation of OSPF has been described in great detail by the standards bodies, the network engineer would do well to become familiar with its internal workings. The following sections describe some of the more common OSPF troubles.

### OSPF Area IDs

When you’re using multiple network area statements under the OSPF configuration, the order of the statements is critical. Check that the networks have been assigned the desired area IDs by checking the output of the **show ip ospf interface** command.

### OSPF Does Not Start

The OSPF process cannot start on a router if a router ID cannot be established. Check the output of **show ip ospf** to see if a router ID has been established. If a router ID has not been established, check to see if the router has an active interface (preferably a loopback interface) with an IP address.

### Verifying Neighbor Relationships

Once a router has been able to start OSPF, it will establish an interface data structure for each interface configured to run OSPF. Check the output of **show ip ospf interface** to ensure that OSPF is active on the intended interfaces. If OSPF is active, check for the parameters described in the section [Section 6.4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch06s04.html "How OSPF Works"). Many OSPF problems may be traced to an incorrectly configured interface.

```
   NewYork#sh ip ospf interface
   ...
   Ethernet0 is up, line protocol is up 
57   **`Internet Address 172.16.1.1/24, Area 0`** 
58   **`Process ID 10, Router ID 172.16.251.1, Network Type BROADCAST, Cost: 10`**
     Transmit Delay is 1 sec, State DR, Priority 1 
     Designated Router (ID) 172.16.251.1, Interface address 172.16.1.1
     No backup designated router on this network
     Timer intervals configured, Hello 10, Dead 40, Wait 40, Retransmit 5
       Hello due in 00:00:02
     Neighbor Count is 0, Adjacent neighbor count is 0 
     Suppress hello for 0 neighbor(s)
   Serial0 is up, line protocol is up 
59   **`Internet Address 172.16.250.1/24, Area 0`** 
60   **`Process ID 10, Router ID 172.16.251.1, Network Type POINT_TO_POINT, Cost: 64`**
     Transmit Delay is 1 sec, State POINT_TO_POINT,
     Timer intervals configured, Hello 10, Dead 40, Wait 40, Retransmit 5
       Hello due in 00:00:01
     Neighbor Count is 1, Adjacent neighbor count is 1 
       Adjacent with neighbor 69.1.1.1
     Suppress hello for 0 neighbor(s)
```

Remember that two routers will not form a neighbor relationship unless the parameters specified in the hello protocol match.

```
NewYork#show ip ospf neighbor

Neighbor ID     Pri   State           Dead Time   Address         Interface
192.168.1.2      1   FULL/  -        00:00:31    172.16.250.2    Serial0
192.168.1.3      1   FULL/  -        00:00:32    172.16.251.2    Serial1
```

If two routers have not been able to establish a neighbor relationship and both are active on the multi-access network (i.e., they are able to ping each other), it is likely that their hello parameters do not match. Use the **debug ip ospf adjacency** command to get details on hello parameter mismatches.

### Route Summarization

If an area has multiple ABRs and one ABR announces more specific routes than the others, all the traffic will flow to that router. This is good if this is the desired effect. Otherwise, if you intend to use all ABRs equally, all ABRs must have identical summary statements.

### Overloaded Routers

The design engineer should be familiar with OSPF -- ABRs do more work than internal routers, and DRs/BDRs do more work than other routers. A router that becomes the DR/BDR on multiple networks does even more work. Routers in stub areas and NSSA areas do less work.

### SPF Overrun

To check the number of times the SPF algorithm has executed, use the command **show ip ospf**. A flapping interface may result in frequent executions of the SPF algorithm that, in turn, may take CPU time away from other critical router processes.

```
   NewYork#sh ip ospf 
61  **`Routing Process "ospf 10" with ID 172.16.251.1`**    
    Supports only single TOS(TOS0) routes
62  **`SPF schedule delay 5 secs, Hold time between two SPFs 10 secs`**     
    Number of DoNotAge external LSA 0
    Number of areas in this router is 1. 1 normal 0 stub 0 nssa
       Area BACKBONE(0)
       Number of interfaces in this area is 3
       Area has no authentication
63     **`SPF algorithm executed 24 times`**
       Area ranges are
       Link State Update Interval is 00:30:00 and due in 00:11:48
       Link State Age Interval is 00:20:00 and due in 00:11:48
       Number of DCbitless LSA 1
       Number of indication LSA 0
       Number of DoNotAge LSA 0
```

In this example, the SPF algorithm has been executed 24 times since the router was rebooted (line 63). Note that SPF is scheduled to delay its execution for 5 seconds after the receipt of an LSA update and the minimum time between SPF executions is set to 10 seconds (line 61). This keeps SPF from using up all the processor resources in the event that an interface is flapping.

To change these timers, use the following command under the OSPF configuration:

```
timers spf <_`schedule delay in seconds`_> <_`hold-time in seconds`_>
```
### Using the LS Database

Since the LS database is the input to the SPF algorithm, you can analyze it to troubleshoot missing routes. Analyzing the LS database can be particularly useful when you’re working with stub areas, totally stubby areas, or NSSAs, since these areas block certain LSAs.

The output of **show ip ospf database database-summary** is a useful indicator of the size of the LS database and its components. The command **show ip ospf database** shows the header information from each LSA.

### Network Logs

The output of the command **show log** contains useful historical data and may be used to analyze a network outage.

### Debug Commands

The most useful **debug** commands are **debug ip ospf adjacency** and **debug ip ospf events** . These commands are useful in troubleshooting neighbor relationships. Other **debug** commands available are **debug ip ospf flood**, **debug ip ospf lsa-generation**, **debug ip ospf packet**, **debug ip ospf retransmission**, **debug ip ospf spf**, and **debug ip ospf tree**.

## Summing Up

OSPF can support very large networks -- the OSPF hierarchy allows almost unlimited growth because new areas can be added to the network without impacting other areas. Dijkstra’s SPF algorithm leads to radical improvements in convergence time, and OSPF does not suffer from the routing loop issues that DV protocols manifest.

OSPF exhibits all the advantages of a classless routing protocol. Variable Length Subnet Masks permit efficient use of IP addresses. Discontiguous networks can be supported since LSAs carry subnet mask information, and routes can be summarized at arbitrary bit boundaries. Summarization reduces routing protocol overhead and simplifies network management.

Furthermore, OSPF does not tie up network bandwidth and CPU resources in periodic routing updates. Only small hello packets are transmitted on a regular basis.

These OSPF benefits come at a price:

- OSPF is a complex protocol requiring a structured topology. A haphazard environment, without a plan for network addresses, route summarization, LS database sizes, and router performance, will yield a real mess.
    
- A highly trained staff is required to engineer and operate a large OSPF network.
    
- OSPF maintains an LS database that requires sizeable memory, and the SPF algorithm can hog CPU resources if the size of the topology database has grown out of bounds. Splitting an area to reduce the size of the LS database may not be straightforward, depending on the topology of the area.
    
- OSPF assumes a hierarchical network topology -- migrating a network from another protocol to OSPF requires extensive planning.
