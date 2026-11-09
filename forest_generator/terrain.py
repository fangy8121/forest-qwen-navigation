"""原 perlinCompound 公式和原 1 m、反对角线三角地面。"""
import numpy as np
class Perlin:
    def __init__(self, rng):
        self.p = np.tile(rng.permutation(256), 2)

    def noise(self, x, y, z=0.): #定义
        x, y, z = np.broadcast_arrays(x, y, z) #
        ix, iy, iz = [np.floor(a).astype(int) & 255 for a in (x,y,z)]
        x,y,z = [a-np.floor(a) for a in (x,y,z)]
        fade = lambda a: a*a*a*(a*(a*6-15)+10)
        lerp = lambda t,a,b: a+t*(b-a) 
        def grad(h,x,y,z):
            h=h&15
            u=np.where(h<8,x,y);v=np.where(h<4,y,np.where((h==12)|(h==14),x,z))
            return np.where(h&1,-u,u)+np.where(h&2,-v,v)
        p=self.p; a=p[ix]+iy; b=p[ix+1]+iy
        aa=p[a]+iz;ab=p[a+1]+iz;ba=p[b]+iz;bb=p[b+1]+iz 
        u,v,w=fade(x),fade(y),fade(z)
        return (1+lerp(w,lerp(v,lerp(u,grad(p[aa],x,y,z),grad(p[ba],x-1,y,z)),
            lerp(u,grad(p[ab],x,y-1,z),grad(p[bb],x-1,y-1,z))),
            lerp(v,lerp(u,grad(p[aa+1],x,y,z-1),grad(p[ba+1],x-1,y,z-1)),
            lerp(u,grad(p[ab+1],x,y-1,z-1),grad(p[bb+1],x-1,y-1,z-1)))))/2


class Terrain:
    def __init__(self, config, size, rng):
        self.config=config
        self.noise=Perlin(rng)
        self.slope=rng.uniform(-config['slope_range'],config['slope_range'],2)
        self.amplitude2=float(rng.uniform(0,config['amplitude2']))
        # 噪声相位 offset 与地面留白分开；旧配置仍使用原来的不对称边界。
        padding=config.get('padding')
        x0,y0=(-config['offset'],-config['offset']) if padding is None else (-padding,-padding)
        x1,y1=(size[0]+7,size[1]+2) if padding is None else (size[0]+padding,size[1]+padding)
        self.xs=np.arange(x0,x1+config['resolution']/2,config['resolution'])
        self.ys=np.arange(y0,y1+config['resolution']/2,config['resolution'])
        xx,yy=np.meshgrid(self.xs,self.ys,indexing='ij')
        self.heights=self.compound(xx,yy)
        self.vertices=np.c_[xx.ravel(),yy.ravel(),self.heights.ravel()]
        i,j=np.meshgrid(np.arange(len(self.xs)-1),np.arange(len(self.ys)-1),indexing='ij')
        a=(i*len(self.ys)+j).ravel();b=a+len(self.ys)
        self.faces=np.vstack([np.c_[a,b,a+1],np.c_[b+1,a+1,b]])

    def compound(self,x,y):
        c=self.config
        return (c['amplitude']*self.noise.noise((x+c['offset'])*c['frequency'],(y+c['offset'])*c['frequency'])+
            self.amplitude2*self.noise.noise(x*c['frequency2']+1245,y*c['frequency2']+1245,54)+
            self.slope[0]*x+self.slope[1]*y)

    def __call__(self, points):
        """实际三角地面的高度；让树根和轨迹贴合网格，而非穿插在噪声曲面之间。"""
        return grid_height(points,self.xs,self.ys,self.heights)


def grid_height(points,xs,ys,heights):
    q=(np.asarray(points)[...,:2]-[xs[0],ys[0]])/(xs[1]-xs[0])
    ij=np.floor(q).astype(int)
    ij=np.clip(ij,[0,0],np.array(heights.shape)-2)
    u,v=(q-ij).T;i,j=ij.T
    a,b,c,d=heights[i,j],heights[i+1,j],heights[i,j+1],heights[i+1,j+1]
    return np.where(u+v<=1,a+(b-a)*u+(c-a)*v,d+(c-d)*(1-u)+(b-d)*(1-v))
