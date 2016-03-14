
class Pmf(object):
    def __init__(self):
        self.d = {}

    def set(self,x,y=0):
        self.d[x] = y

    def inc(self,x,term=1):
        self.d[x] = self.d.get(x,0) + term

    def mult(self,x,factor):
        self.d[x] = self.d.get(x,0) * factor

    def remove(self,x):
        del self.d[x]

    def total(self):
        return sum(self.d.values())

    def normalize(self):
        factor = 1.0 / self.total()
        for x in self.d:
            self.d[x] *= factor

class Cookie(Pmf):
    mixes = {
        'a': { 'c' : 0.75, 'd' :0.25 },
        'b': { 'c' : 0.5,  'd' : 0.5 },
    }

    def __init__(self,hypos):
        super(Cookie,self).__init__()
        for hypo in hypos:
            self.set(hypo,1)
        self.normalize()

    def update(self,data):
        for hypo in self.d.keys():
            likely = self.likehood(data,hypo)
            self.mult(hypo,likely)
        self.normalize()

    def likehood(self,data,hypo):
        return self.mixes[hypo][data]

if __name__ == '__main__':
    hypos = [ 'a', 'b' ]
    c = Cookie(hypos)
    c.update('c')
    for x in c.d:
        print x,c.d[x]
