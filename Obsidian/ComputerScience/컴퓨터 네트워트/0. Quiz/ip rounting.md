
---
# Chapter 2. Routing Information Protocol (RIP)

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

Consider a simple network with two routers connected to each other ([Figure 2-5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02s03.html#iprouting-CHP-2-FIG-5 "Figure 2-5. Split horizon")).

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

## Subnet Masks

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

## Route Summarization

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

## Default Route

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

## Fine-Tuning RIP

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

## Summing Up

RIP is a relatively simple protocol, easy to configure and very reliable. The robustness of RIP is evident from the fact that various implementations of RIP differ in details and yet work well together. A standard for RIP wasn’t put forth until 1988 (by Charles Hedrick, in RFC 1058). Small, homogeneous networks are a good match for RIP. However, as networks grow, other routing protocols may look more attractive for several reasons:

- The RIP metric does not account for link bandwidth or delay.
    
- The exchange of full routing updates every 30 seconds does not scale for large networks -- the overhead of generating and processing all routes can be high.
    
- RIP convergence times can be too long.
    
- Subnet mask information is not exchanged in RIP updates, so Variable Length Subnet Masks are not supported.
    
- The RIP metric restricts the network diameter to 15 hops.

# Chapter 3. Interior Gateway Routing Protocol (IGRP)

The second Distance Vector protocol that we will examine is the Interior Gateway Routing Protocol, or IGRP. IGRP and RIP are close cousins: both are based on the Bellman-Ford Distance Vector (DV) algorithms. DV algorithms propagate routing information from neighbor to neighbor; if a router receives the same route from multiple neighbors, it chooses the route with the lowest metric. All DV protocols need robust strategies to cope with _bad_ routing information. Bad routes can linger in a network when information about the loss of a route does not reach some router (for instance, because of the loss of a route update packet), which then inserts the bad route back into the network. IGRP uses the same convergence strategies as RIP: triggered updates, route hold-downs, split horizon, and poison reverse.

IGRP has been widely deployed in small to mid-sized networks because it can be configured with the same ease as RIP, but its metric represents bandwidth and delay, in addition to hop count. The ability to discriminate between paths based on bandwidth and delay is a major improvement over RIP.

IGRP is a Cisco proprietary protocol; other router vendors do not support IGRP. Keep this in mind if you are planning a multivendor router environment.

The following section gets us started with configuring IGRP.

## Getting IGRP Running

TraderMary’s network, shown in [Figure 3-1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03.html#iprouting-CHP-3-FIG-1 "Figure 3-1. TraderMary’s network"), can be configured to run IGRP as follows.

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

Getting IGRP started is fairly straightforward. However, if you compare the routing tables in this section to those in the previous chapter on RIP, there is no difference in the next-hop information. More importantly, the route from _NewYork_ to network `172.16.100.0` is still over the direct 56-kbps path rather than the two-hop T-1 path. The two-hop T-1 path is better than the one-hop 56-kbps link. As an example, take a 512-byte packet; it would take 73 ms to copy this packet over a 56-kbits/s link versus 5 ms over two T-1 links. Our expectation is that IGRP should install this two-hop T-1 path, since IGRP has been touted for its metric that includes link bandwidth and delay. [Section 3.2.2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-SECT-2.2 "IGRP Metric") explains why IGRP installs the slower path. [Section 3.2.2.6](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-SECT-2.2.6 "Modifying IGRP metrics") leads us through the configuration changes required to make IGRP install the faster path.

A key difference in this configuration is that, unlike in RIP, each IGRP process is identified by an autonomous system (AS) number. AS numbers are described in detail in the next section.

## How IGRP Works

Since IGRP is such a close cousin of RIP, we will not repeat the details of how DV algorithms work, how updates are sent, and how route convergence is achieved. However, because IGRP employs a much more comprehensive metric, I’ll discuss the IGRP metric in detail. I’ll begin this discussion with AS numbers.

### IGRP Autonomous System Number

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

### IGRP Metric

The RIP metric was designed for small, homogenous networks. Paths were selected based on the number of hops to a destination; the lowest hop-count path was installed in the routing table. IGRP is designed for more complex networks. Cisco’s implementation of IGRP allows the network engineer to customize the metric based on bandwidth, delay, reliability, load, and MTU. In order to compare metrics between paths and select the least-cost path, IGRP converts bandwidth, delay, reliability, delay, and MTU into a scalar quantity -- a _composite_ metric that expresses the desirability of a path. Just as in the case of RIP, a path with a lower composite metric is preferred to a path with a higher composite metric.

The computation of the IGRP composite metric is user-configurable; i.e., the network administrator can specify parameters in the _formula_ used to convert bandwidth, delay, reliability, load, and MTU into a scalar quantity.

The following sections define bandwidth, delay, reliability, load, and MTU. We will then see how these variables can be used to compute the composite metric for a path.

#### Interface bandwidth, delay, reliability, load, and MTU

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

#### Modifying interface bandwidth, delay, and MTU

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

#### IGRP routing update

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

#### Path bandwidth, delay, reliability, load, and MTU

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
#### IGRP composite metric

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

#### Modifying IGRP metrics

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

#### Processing IGRP updates

The processing of IGRP updates is very similar to the processing of RIP updates, described in [Chapter 2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02.html "Chapter 2. Routing Information Protocol (RIP)"). The IGRP update comes with an autonomous system number. If this does not match the IGRP AS number configured on the router receiving the update, the entire upgrade is disregarded. Thus, routers _NewYork_ and _Nairobi_ in TraderMary’s network will receive updates from each other but will discard them.

Each network number received in the update is checked for validity. Illegal network numbers such as `0.0.0.0/8`, `127.0.0.0/8`, and `128.0.0.0/16` are sometimes referred to as “Martian Network Numbers” and will be disregarded when received in an update (RFCs 1009, 1122).

The rules for processing IGRP updates are:

1. If the destination network number is unknown to the router, install the route using the source IP address of the update (provided the route is not indicated as unreachable).
    
2. If the destination network number is known to the router but the update contains a smaller metric, modify the routing table entry with the new next hop and metric.
    
3. If the destination network number is known to the router but the update contains a larger metric, ignore the update.
    
4. If the destination network number is known to the router and the update contains a higher metric that is from the same next hop as in the table, update the metric.
    
5. If the destination network number is known to the router and the update contains the same metric from a different next hop, install the route as long as the maximum number of paths to the same destination is not exceeded. These parallel paths are then used for load balancing. Note that the default maximum number of paths to a single destination is six in IOS Releases 11.0 or later.
    

### Parallel Paths

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

### Steady State

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
 
## Speeding Up Convergence

Like RIP, IGRP implements hold-downs, split horizon, triggered updates, and poison reverse (see [Chapter 2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch02.html "Chapter 2. Routing Information Protocol (RIP)") for details on these convergence methods). Like RIP, IGRP also maintains an update timer, an invalid timer, a hold-down timer, and a flush timer for every route in the routing table:

Update timer (default value: 90 seconds)
	After sending a routing update, IGRP sets the update timer to 0. When the timer expires, IGRP issues another routing update.

Invalid timer (default value: 270 seconds)
	Every time a router receives an update for a route, it sets the invalid timer to 0. The expiration of the invalid timer indicates that the source of the routing information is suspect. Even though the route is declared invalid, packets are still forwarded to the next hop specified in the routing table. Note that prior to the expiration of the invalid timer, IGRP would process any updates received by updating the route’s timers.

Hold-down timer (default value: 280 seconds)
	When the invalid timer expires, the route automatically enters the hold-down phase. During hold-down all updates regarding the route are disregarded -- it is assumed that the network may not have converged and that there may be bad routing information circulating in the network. The hold-down timer is started when the invalid timer expires.

Flush timer (default value: 630 seconds)
	Every time a router receives an update for a route, it sets the flush timer to 0. When the flush timer expires, the route is removed from the routing table and the router is ready to receive a new route update. Note that the flush timer overrides the hold-down timer.

### Setting Timers

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

### Disabling IGRP Hold-Downs

IGRP hold-downs can be disabled with the command:

```
router igrp 10
no metric holddown
```

thus speeding up convergence when a route fails. However, the problem with turning off hold-downs is that if a triggered update regarding the failure does not reach some router, that router could insert bad routing information into the network. Doesn’t this seem like a dangerous thing to do?

Split horizon, triggered updates, and poison reverse are implemented in IGRP much like they are in RIP.

## Route Summarization

IGRP summarizes network numbers when crossing a major network-number boundary, just like RIP does. Route summarization reduces the number of routes that need to be exchanged, processed, and stored.

However, route summarization does not work well in discontiguous networks. Consider the discontiguous network in [Figure 3-5](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s04.html#iprouting-CHP-3-FIG-5 "Figure 3-5. Contiguous and discontiguous networks"). Router _X_ will receive advertisements for `10.0.0.0` from both routers _A_ and _B_. If _X_ sent packets with the destination `10.1.1.1` to _B_, the packet would be lost -- _B_ would have to drop the packet because it would not have a route for `10.1.1.1` in its table. Likewise, if _X_ sent packets with the destination `10.2.1.1` to _A_, the packet would be lost -- _A_ would have to drop the packet because it would not have a route for `10.2.1.1`.

![[Pasted image 20241110172649.png]]
Figure 3-5. Contiguous and discontiguous networks

Both IGRP and RIP networks must be designed in contiguous blocks of major network numbers.

## Default Routes

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

### Multiple Default Routes

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

## Classful Route Lookups

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

## Summing Up

IGRP has the robustness of RIP but adds a major new feature -- route metrics based on bandwidth and delay. This feature -- along with the ease with which it can be configured and deployed -- has made IGRP tremendously popular for small to mid-sized networks. However, IGRP does not address several problems that also affect RIP:

- The exchange of full routing updates does not scale for large networks -- the overhead of generating and processing all routes in the AS can be high.
    
- IGRP convergence times can be too long.
    
- Subnet mask information is not exchanged in IGRP updates, so Variable Length Subnet Masks (VLSM) and discontiguous address spaces are not supported.
    

These issues may be too significant to overlook in large IP networks in which address-space conservation may necessitate VLSM, full route updates would be so large that they would consume significant network resources (serial links to branches tend to saturate quickly, and smaller routers may consume a lot of CPU power just to process all the routes at every update interval), and the convergence times may be too long because of the network diameter. Even small to mid-sized networks may choose not to implement IGRP if convergence time is an issue.

---
The definition of small, medium, and large IP networks can be discussed ad nauseam because of the number of variables involved (number of routers and routes, network bandwidth/utilization, network delay/latency, etc.), but rough measures are as follows: small -- a few dozen routers with up to a few hundred routes; medium -- a few hundred routers with a few thousand routes; large -- anything bigger than medium.

# Chapter 4. Enhanced Interior Gateway Routing Protocol (EIGRP)

The Enhanced Interior Gateway Routing Protocol (EIGRP), referred to as an advanced Distance Vector protocol, offers radical improvements over IGRP. Traditional DV protocols such as RIP and IGRP exchange periodic routing updates with all their neighbors, saving the best distance (or metric) and the vector (or next hop) for each destination. EIGRP differs in that it saves not only the best (least-cost) route but all routes, allowing convergence to be much quicker. Further, EIGRP updates are sent only upon a network topology change; updates are not periodic.

Getting EIGRP running is not much more difficult than getting IGRP running, as we will see in Section 4.1.

Even though EIGRP offers radical improvements over IGRP, there are similarities between the protocols. Like IGRP, EIGRP bases its metric on bandwidth, delay, reliability, load, and MTU (see Section 4.2).

The fast convergence feature in EIGRP is due to the Diffusing Update Algorithm (DUAL), discussed in Section 4.3.

EIGRP updates carry subnet mask information. This allows EIGRP to summarize routes on arbitrary bit boundaries, support classless route lookups, and allow the support of Variable Length Subnet Masks (VLSM). This is discussed in Section 4.4 and Section 4.5.

Setting up default routes in EIGRP is discussed in [Section 4.6](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s06.html "Default Routes").

Troubleshooting EIGRP can be tricky. This chapter ends with some troubleshooting tips in [Section 4.7](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s07.html "Troubleshooting EIGRP").

EIGRP is a Cisco proprietary protocol; other router vendors do not support EIGRP. Keep this in mind if you are planning a multivendor router environment.

This chapter focuses on EIGRP’s enhancements over IGRP: the use of DUAL; and the use of subnet masks in updates, which in turn allow VLSM and route summarization at arbitrary bit boundaries. This chapter does not cover router metrics in detail or the concept of parallel paths. Those concepts have not changed much in EIGRP. I assume that the reader is familiar with IGRP.

# Getting EIGRP Running

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

Each EIGRP process is identified by an autonomous system (AS) number, just like IGRP processes. Routers with the _same_ AS numbers will exchange routing information with each other, resulting in a _routing domain_ . Routers with dissimilar AS numbers will not exchange any routing information by default. However, routes from one routing domain can be leaked into another domain through the redistribution commands -- this is covered in [Chapter 8](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch08.html "Chapter 8. Administrative Controls").

Compare the routing table in this section with the corresponding table for IGRP in [Chapter 3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03.html "Chapter 3. Interior Gateway Routing Protocol (IGRP)"). The essential contents are identical: the same routes with the same next hops. However, the route metrics look much bigger and the route update times are very high. IGRP routes would have timed out a while ago.

EIGRP metrics are essentially derived from IGRP metrics. The following section provides a quick summary.

# EIGRP Metric

The EIGRP composite metric is computed exactly as the IGRP metric is and then multiplied by 256. Thus, the default expression for the EIGRP composite metric is:

_Metric =_ [_BandW +Delay_]x 256

where _BandW_ and _Delay_ are computed exactly as for IGRP (see [Section 3.2.2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-SECT-2.2 "IGRP Metric") in [Chapter 3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03.html "Chapter 3. Interior Gateway Routing Protocol (IGRP)")). In summary, _BandW_ is computed by taking the smallest bandwidth (expressed in kbits/s) from all outgoing interfaces to the destination (including the destination) and dividing 10,000,000 by this number (the smallest bandwidth), and _Delay_ is the sum of all the delay values to the destination network (expressed in tens of microseconds).

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

Just like IGRP, EIGRP can be made to use load and reliability in its metric by modifying the parameters k1, k2, k3, k4, and k5 (see [Section 3.2.2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03s02.html#iprouting-CHP-3-SECT-2.2 "IGRP Metric") in [Chapter 3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03.html "Chapter 3. Interior Gateway Routing Protocol (IGRP)")).

The constants k1, k2, k3, k4, and k5 can be modified with the following command:

```
metric weights tos k1 k2 k3 k4 k5
```
### Warning

Cisco strongly recommends _not_ modifying the k1, k2, k3, k4, and k5 values for EIGRP.

# How EIGRP Works

Unlike traditional DV protocols such as RIP and IGRP, EIGRP does not rely on _periodic_ updates: routing updates are sent only when there is a change. Remember that RIP and IGRP reset the invalid and flush timers upon receiving a route update. When a route is lost, the updates stop; the invalid and flush timers grow and grow (the timers are not reset), and, ultimately, the route is flushed from the routing table. This process of convergence assumes periodic updates. EIGRP’s approach has the advantage that network resources are not consumed by periodic updates. However, if a router dies, taking away all its downstream routes, how would EIGRP detect the loss of these routes? EIGRP relies on small _hello packets_ to establish neighbor relationships and to detect the loss of a neighbor. Neighbor relationships are discussed in detail in the next section.

RIP and IGRP suffer from a major flaw: _routing loops_ . Routing loops happen when information about the loss of a route does not reach all routers in the network because an update packet gets dropped or corrupted. These routers (that have not received the information about the loss of the route) inject bad routing information back into the network by telling their neighbors about the route they know. EIGRP uses _reliable_ transmission for all updates between neighbors. Neighbors acknowledge the receipt of updates, and if an acknowledgment is not received, EIGRP retransmits the update.

RIP and IGRP employ a battery of techniques to reduce the likelihood of routing loops: split horizon, hold-down timers, and poison reverse. These techniques do not guarantee that loops will not occur and, in any case, result in long convergence times. EIGRP uses the Diffusing Update Algorithm (DUAL) for all route computations. DUAL’s convergence times are an order of magnitude lower than those of traditional DV algorithms. DUAL is able to achieve such low convergence times by maintaining a table of loop-free paths to every destination, in addition to the least-cost path. DUAL is described in more detail later in this chapter.

DUAL can support IP, IPX, and AppleTalk. A protocol-dependent module encapsulates DUAL messages and handles interactions with the routing table. In summary, DUAL requires:

1. A method for the discovery of new neighbors and their loss (see the next section, [Section 4.3.1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.1 "Neighbor Relationship")).
    
2. Reliable transmission of update packets between neighbors (see the later section [Section 4.3.2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.2 "Reliable Transport Protocol")).
    
3. Protocol-dependent modules that can encapsulate DUAL traffic in IP, IPX, or AppleTalk. This text will deal only with EIGRP in IP networks (see the later section [Section 4.3.4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.4 "Protocol-Dependent Module")).
    

I’ll end this section with a discussion of EIGRP packet formats.

## Neighbor Relationship

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

## Reliable Transport Protocol

The EIGRP transport mechanism uses a mix of multicast and unicast packets, using reliable delivery when necessary. All transmissions use IP with the protocol type field set to 88. The IP multicast address used is `224.0.0.10`.

DUAL requires guaranteed and sequenced delivery for some transmissions. This is achieved using acknowledgments and sequence numbers. So, for example, _update packets_ (containing routing table data) are delivered reliably (with sequence numbers) to all neighbors using multicast. _Acknowledgment packets_ _--_ with the correct sequence number -- are expected from every neighbor. If the correct acknowledgment number is not received from a neighbor, the update is retransmitted as a unicast.

The sequence number (seq num) in the last packet from the neighbor is recorded to ensure that packets are received in sequence. The number of packets in the queue that might need retransmission is shown as a queue count (QCnt), and the smoothed round trip time (SRTT) is used to estimate how long to wait before retransmitting to the neighbor. The retransmission timeout (RTO) is the time the router will wait for an acknowledgment before retransmitting the packet in the queue.

Some transmissions do not require reliable delivery. For example, hello packets are multicast to all neighbors on an Ethernet segment, whereas acknowledgments are unicast. Neither hellos nor acknowledgments are sent reliably.

EIGRP also uses _queries_ and _replies_ as part of DUAL. Queries are multicast or unicast using reliable delivery, whereas replies are always reliably unicast. Query and reply packets are discussed in more detail in the next section.

## Diffusing Update Algorithm (DUAL)

All route computations in EIGRP are handled by DUAL. One of DUAL’s tasks is maintaining a table of loop-free paths to every destination. This table is referred to as the _topology table_ . Unlike traditional DV protocols that save only the best (least-cost) path for every destination, DUAL saves all paths in the topology table. The least-cost path(s) is copied from the topology table to the routing table. In the event of a failure, the topology table allows for very quick convergence if another loop-free path is available. If a loop-free path is not found in the topology table, a route recomputation must occur, during which DUAL queries its neighbors, who, in turn, may query their neighbors, and so on... hence the name “Diffusing” Update Algorithm.

These processes are described in detail in the following sections.

### Reported distance

Just like RIP and IGRP, EIGRP calculates the lowest cost to reach a destination based on updates[[4](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#ftn.ch04-FTNOTE-1)] from neighbors. An update from a router _R_ contains the cost to reach the destination network _N_ from _R_. This cost is referred to as the _reported distance_ (RD). _NewYork_ receives an update from _Ames_ with a cost of 281,600, which is _Ames_’s cost to reach `172.16.100.0`. In other words, the RD for _Ames_ to reach `172.160.100.0` as reported to _NewYork_ is 281,600. Just like _Ames_, _Chicago_ will report its cost to reach `172.16.100.0`. _Chicago_’s RD is 2,195,456 (see [Figure 4-2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-2 "Figure 4-2. Ames is a feasible successor for 172.16.100.0")).

![[Pasted image 20241110174908.png]]
Figure 4-2. Ames is a feasible successor for 172.16.100.0

### Feasible distance and successor

_NewYork_ will compute its cost to reach `172.16.100.0` via _Ames_ and _Chicago_. _NewYork_ will then compare the metrics for the two paths. _NewYork_’s cost via _Ames_ is 46,251,776. _NewYork_’s cost via _Chicago_ is 2,707,456. The lowest cost to reach a destination is referred to as the _feasible distance_ (FD) for that destination. _NewYork_’s FD to `172.16.100.0` is 2,707,456 (_BandW_ = 1,544 and _Delay_ = 4,100). The next-hop router in the lowest-cost path to the destination is referred to as the _successor_ . _NewYork_’s successor for `172.16.100.0` is `172.16.50.1` (_Chicago_).

### Feasibility condition and feasible successor

If a reported distance for a destination is less than the feasible distance for the same destination, the router that advertised the RD is said to satisfy the _feasibility condition_ (FC) and is referred to as a _feasible successor_ (FS). _NewYork_ sees an RD of 281,600 via _Ames_, which is lower than _NewYork_’s FD of 2,707,456. _Ames_ satisfies the FC. _Ames_ is an FS for _NewYork_ to reach `172.16.100.0`.

### Loop freedom

The feasibility condition is a test for _loop freedom_ : if the FC is met, the router advertising the RD must have a path to the destination not through the router checking the FC -- if it did, the RD would have been higher than the FD.

Let’s illustrate this concept with another example. Consider the network in [Figure 4-3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-3 "Figure 4-3. Loop freedom"). The metric values used in this example have been simplified to small numbers to make it easier to follow the concept.

![[Pasted image 20241110174948.png]]
Figure 4-3. Loop freedom

Router _A_’s best route to network _N_ is via router _B_, and the cost of this path is 100 (_A_’s FD to _N_ is 100). Router _X_ also knows how to get to network _N_; _X_ advertises _N_ to _A_ in an update packet (_A_ copies this information into its topology table). In the event that _A_’s link to _B_ fails, _A_ can use the route to _N_ via _X_ if _X_ does not use _A_ to get to _N_ (in other words, if the path is loop-free). Thus, the key question for _A_ to answer is whether or not the path that _X_ advertises is loop-free.

Here is how _A_ answers this question. Let’s say that _X_ advertises _N_ with a metric of 90 (_X_’s RD for _N_). _A_ compares 90 (RD) with 100 (FD). Is RD < FD? This comparison is the FC check. Since _A_’s FD is 100, _X_’s path to _N_ must not be via _A_ (and is loop-free). If _X_ advertises _N_ with a metric of 110, _X_’s path to _N_ could be via _A_ (the RD is not less than the FD, so the FC check fails) -- 110 could be _A_’s cost added to the metric of the link between _A_ and _X_ (and, hence, is not guaranteed to be free of a loop).

### Topology table

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

### Convergence in DUAL -- local computation

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

### Convergence in DUAL -- diffusing computation

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

## Protocol-Dependent Module

The successors in the DUAL topology table are eligible for installation in the routing table. Successors represent the best path to the destination known to DUAL. However, whether the successor is copied into the routing table is another matter. The router may be aware of a route to the same destination from another source (such as another routing protocol or via a static route) with a lower _distance_. The IP protocol-dependent module (PDM) handles this task. The PDM may also carry information in the reverse direction -- from the routing table to the topology table. This will occur if routes are being redistributed into EIGRP from another protocol.

The PDM is also responsible for encapsulating EIGRP messages in IP packets.

## EIGRP Packet Format

EIGRP packets are encapsulated directly in IP with the protocol field set to 88. The destination IP address in EIGRP depends on the packet type -- some packets are sent as multicast (with an address of `224.0.0.10`) and others are sent as unicast (see the earlier section [Section 4.3.2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.2 "Reliable Transport Protocol") for more details). The source IP address is the IP address of the interface from which the packet is issued.

Following the IP header is an EIGRP header. Key fields in the EIGRP header are as follows, and are also shown in [Figure 4-6](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-6 "Figure 4-6. Format of EIGRP packets"):

- The _opcode_ field specifies the EIGRP packet type (update, query, reply, hello).
    
- The _checksum_ applies to the entire EIGRP packet, excluding the IP header.
    
- The rightmost bit in the _flags_ field is the initialization bit and is used in establishing a new neighbor relationship (see [Section 4.3.1](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.1 "Neighbor Relationship") earlier in this chapter).
    
- The _sequence_ and _ack_ fields are used to send messages reliably (see [Section 4.3.2](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-SECT-3.2 "Reliable Transport Protocol") earlier in this chapter).
    
- The _AS number_ identifies the EIGRP process issuing the packet. The EIGRP process receiving the packet will process the packet only if the receiving EIGRP process has the same AS number; otherwise, the packet will be discarded.
    

![[Pasted image 20241110175413.png]]
Figure 4-6. Format of EIGRP packets

The fields following the EIGRP header depend on the opcode field. Of particular interest to routing engineers is the information in updates. We will ignore the other types of EIGRP messages and focus on IP internal route updates and IP external route updates.

_Internal_ routes contain destination network numbers learned within this EIGRP AS. For example, _NewYork_ learns `172.16.50.0` from EIGRP 10 on _Chicago_ as an internal route.

_External_ routes contain destination network numbers that were not learned within this EIGRP AS but rather derived from another routing process and redistributed into this EIGRP AS.

Internal and external routes are represented differently in the EIGRP update.

### Internal routes

Internal routes have a _type_ field of 0x0102. The metric information contained with the route is much like IGRP’s (see [Chapter 3](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch03.html "Chapter 3. Interior Gateway Routing Protocol (IGRP)")). However, there are two new fields: _next hop_ and _prefix length_. [Figure 4-7](https://learning.oreilly.com/library/view/ip-routing/0596002750/ch04s03.html#iprouting-CHP-4-FIG-7 "Figure 4-7. EIGRP internal route") shows the value field for the IP internal route.

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

### External routes

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
