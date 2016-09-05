import collections
from math import log

def uniqueCounts(items):
    results = collections.defaultdict(int)
    for item in items:
        results[item] +=1
    return results


def entropy(items):
   log2=lambda x:log(x)/log(2)
   results=uniqueCounts(items)
   #Now calculate the entropy
   ent=0.0
   for r in results.keys():
      p=float(results[r])/len(items)
      ent=ent-p*log2(p)
   return ent

def giniimpurity(items):
    counts = uniqueCounts(items=items)
    dataLength = len(items)
    imp = 0.0
    for name, count in counts.items():
        p1 = float(count)/dataLength
        for name2,count2 in counts.items():
            if name == name2:
                continue
            p2 = float(count2)/dataLength
            imp += (p1 * p2)
    return imp


