# Round 2

- [Round 2](#round-2)
  - [Introduction](#introduction)
    - [Algorithmic challenge](#algorithmic-challenge)
    - [Manual challenge](#manual-challenge)
  - [Initial analysis](#initial-analysis)
    - [**Algorithmic trading challenge**](#algorithmic-trading-challenge)
    - [**Manual trading challenge**](#manual-trading-challenge)

## Introduction

### Algorithmic challenge

For this round, we were given a new asset to trade algorithmically: **orchids**.
Orchids are "imported" from a different island and therefore incurr

### Manual challenge

For the manual trading challenge, we were given 2'000'000 shells to trade, and were given a set of exchange rates to trade between three different assets: snowballs, pizza, and wasabi roots.

## Initial analysis

---

### **Algorithmic trading challenge**

**Approach**
[to be filled]

### **Manual trading challenge**

**Approach**
For the manual trading challenge, I immediately noticed that it was a directed graph problem and we could use some path-finding algorithms in order to find the most profitable path. Because the weights were multiplied and not added, I first calculated the inverse log of the weights in order to convert the problem into a shortest non-negative path problem. However, I realized that since there were only 4 nodes ($hells, pizza, wasabi root and snowballs), it would simply be easier to calculate the profit of each possible path and then choose the most profitable one, rather then trying to find a clever solution to it.
