"""局部凸 Bézier 弯道、曲率速度规划与连续运动接入。"""
import numpy as np
from scipy.interpolate import BPoly
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from .geometry import segment_distances


def path_distance(points, polyline):
    return np.minimum.reduce([segment_distances(points,a,b) for a,b in zip(polyline[:-1],polyline[1:])])


def corner_curve(vertex, incoming, outgoing, radius):
    entry=vertex-radius*incoming;exit=vertex+radius*outgoing;handle=.4*radius
    # 控制多边形始终向同一侧转弯；端点的一阶导沿直线，二阶导为零。
    control=np.array([entry,entry+handle*incoming,entry+2*handle*incoming,
                      exit-2*handle*outgoing,exit-handle*outgoing,exit])
    return control,5*handle


class QuinticPath:
    @classmethod
    def from_geometry(cls,geometry,flight):
        """Time a preassembled C2 spatial curve without re-polygonizing it."""
        result=cls.__new__(cls)
        result.geometry=geometry;result.direct=None;result.prefix=None;result.advance=0.
        result._timing(flight.max_speed,flight.max_acceleration,flight.max_jerk,
                       flight.max_yaw_rate,flight.max_yaw_acceleration)
        result._finish()
        return result

    def __init__(self, points, max_speed=1., max_acceleration=1., *, start_vel=None,start_acc=None,
                 max_jerk=4.,max_yaw_rate=1.,max_yaw_acceleration=2.,tangent_scale=1.,
                 corner_distance=1.5,max_path_deviation=.3,check=None):
        points=np.asarray(points,float)
        points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-6]]
        if len(points)<2:
            raise ValueError('path needs two distinct points')
        self.original=points
        self.direct=None;self.prefix=None;self.advance=0.
        self.geometry=self._geometry(points,corner_distance*tangent_scale,max_path_deviation,check)
        self._timing(max_speed,max_acceleration,max_jerk,max_yaw_rate,max_yaw_acceleration)
        v=np.zeros(3) if start_vel is None else np.asarray(start_vel)
        a=np.zeros(3) if start_acc is None else np.asarray(start_acc)
        if np.linalg.norm(v)+np.linalg.norm(a)>1e-7:
            self._splice(v,a,max_speed,max_acceleration,max_jerk,max_yaw_rate,max_yaw_acceleration,max_path_deviation,check)
        self._finish()

    @staticmethod
    def _geometry(points,radius,deviation,check):
        delta=np.diff(points,axis=0);lengths=np.linalg.norm(delta,axis=1);direction=delta/lengths[:,None]
        controls=[];widths=[];last=points[0]
        def line(end):
            nonlocal last
            length=np.linalg.norm(end-last)
            if length>1e-7:
                controls.append(np.linspace(last,end,6));widths.append(length)
            last=end
        for i in range(1,len(points)-1):
            cosine=np.clip(direction[i-1]@direction[i],-1.,1.)
            if cosine>1-1e-7:
                continue
            if cosine<-.98:
                raise ValueError('turn requires braking or replanning')
            limit=min(radius,.45*lengths[i-1],.45*lengths[i])
            selected=None
            # 在局部控制点族中选择最大的可行弯道，优先降低曲率。
            for r in limit*np.array([1.,.8,.6,.4,.25,.12]):
                c,width=corner_curve(points[i],direction[i-1],direction[i],r)
                curve=BPoly(c[:,None,:],[0.,width])
                samples=curve(np.linspace(0,width,max(65,int(width/.02)+1)))
                if path_distance(samples,points[i-1:i+2]).max()>deviation:
                    continue
                if check is None or check(samples):
                    selected=c,width;break
            if selected is None:
                raise ValueError('no clear local turn')
            c,width=selected;line(c[0]);controls.append(c);widths.append(width);last=c[-1]
        line(points[-1])
        return BPoly(np.stack(controls,axis=1),np.r_[0.,np.cumsum(widths)])

    def _base(self,t,jerk=False):
        t=np.clip(t,0.,self.clock.x[-1])
        q=self.clock(t);sd=self.clock(t,nu=1);sdd=self.clock(t,nu=2)
        q=np.clip(q,self.geometry.x[0],self.geometry.x[-1])
        g1=self.geometry(q,nu=1);g2=self.geometry(q,nu=2)
        p=self.geometry(q);v=g1*sd[...,None];a=g2*sd[...,None]**2+g1*sdd[...,None]
        if jerk:
            j=self.geometry(q,nu=3)*sd[...,None]**3+3*g2*(sd*sdd)[...,None]+g1*self.clock(t,nu=3)[...,None]
            return p,v,a,j
        return p,v,a

    @staticmethod
    def _limit_ratio(v,a,j,max_speed,max_acceleration,max_jerk,max_yaw_rate):
        speed=np.linalg.norm(v,axis=-1)
        yaw=np.abs(v[...,0]*a[...,1]-v[...,1]*a[...,0])/np.maximum(np.sum(v[...,:2]**2,axis=-1),1e-8)
        yaw=np.where(speed>.02,yaw,0.)
        return max(float(speed.max()/max_speed),float(np.sqrt(np.linalg.norm(a,axis=-1).max()/max_acceleration)),
                   float(np.cbrt(np.linalg.norm(j,axis=-1).max()/max_jerk)),float(yaw.max()/max_yaw_rate))

    def _timing(self,max_speed,max_acceleration,max_jerk,max_yaw_rate,max_yaw_acceleration):
        q=np.unique(np.concatenate([np.linspace(a,b,max(4,int(np.ceil((b-a)/.08))+1))
                                   for a,b in zip(self.geometry.x[:-1],self.geometry.x[1:])]))
        p=self.geometry(q);g1=self.geometry(q,nu=1);g2=self.geometry(q,nu=2)
        norm=np.linalg.norm(g1,axis=1)
        curvature=np.linalg.norm(np.cross(g1,g2),axis=1)/np.maximum(norm**3,1e-10)
        yaw_curvature=np.abs(g1[:,0]*g2[:,1]-g1[:,1]*g2[:,0])/np.maximum(norm*np.sum(g1[:,:2]**2,axis=1),1e-10)
        speed=np.minimum(max_speed,np.minimum(np.sqrt(.65*max_acceleration/np.maximum(curvature,1e-8)),
                                               max_yaw_rate/np.maximum(yaw_curvature,1e-8)))
        distance=np.linalg.norm(np.diff(p,axis=0),axis=1)
        caps=speed.copy()
        for _ in range(32):
            speed=caps.copy();speed[[0,-1]]=0.
            for i in range(1,len(q)):
                speed[i]=min(speed[i],np.sqrt(speed[i-1]**2+max_acceleration*distance[i-1]))
            for i in range(len(q)-2,-1,-1):
                speed[i]=min(speed[i],np.sqrt(speed[i+1]**2+max_acceleration*distance[i]))
            speed=np.minimum(speed,gaussian_filter1d(speed,2,mode='nearest'))
            qdot=speed/norm
            dt=2*np.diff(q)/np.maximum(qdot[:-1]+qdot[1:],1e-8)
            times=np.r_[0.,np.cumsum(dt)]
            qddot=np.gradient(qdot,times);qddot[[0,-1]]=0.
            self.clock=BPoly.from_derivatives(times,[[x,v,a] for x,v,a in zip(q,qdot,qddot)])
            # 包含每段左右极限；jerk 在连接处可跳变，不能只检查右侧值。
            ts=(times[:-1,None]+dt[:,None]*np.linspace(1e-7,1-1e-7,33)).ravel()
            _,v,a,j=self._base(ts,True)
            actual_speed=np.linalg.norm(v,axis=1)
            speed2=np.maximum(np.sum(v[:,:2]**2,axis=1),1e-8)
            yaw_rate=(v[:,0]*a[:,1]-v[:,1]*a[:,0])/speed2
            ratios=np.maximum.reduce([actual_speed/max_speed,
                np.sqrt(np.linalg.norm(a,axis=1)/max_acceleration),
                np.cbrt(np.linalg.norm(j,axis=1)/max_jerk),np.abs(yaw_rate)/max_yaw_rate])
            yaw_acc=(v[:,0]*j[:,1]-v[:,1]*j[:,0])/speed2-2*yaw_rate*np.sum(v[:,:2]*a[:,:2],axis=1)/speed2
            ratios=np.maximum(ratios,np.sqrt(np.abs(yaw_acc)/max_yaw_acceleration))
            local=ratios.reshape(-1,33).max(axis=1)
            if local.max()<=1.001:
                return
            reduction=np.maximum(np.r_[local,1.],np.r_[1.,local])
            # 将降速量平滑分配到邻域，避免单个速度节点下压后制造新的 jerk 尖峰。
            reduction=np.where(reduction>1.001,reduction*1.04,1.)
            reduction=gaussian_filter1d(maximum_filter1d(reduction,11),3)
            caps/=reduction
        raise ValueError('trajectory timing did not converge')

    def _splice(self,v,a,max_speed,max_acceleration,max_jerk,max_yaw_rate,max_yaw_acceleration,deviation,check):
        base_times=np.linspace(0.,self.clock.x[-1],300)
        positions=self._base(base_times)[0]
        reach=max(.3,np.linalg.norm(v)**2/max_acceleration+.2)
        candidates=np.flatnonzero(np.linalg.norm(positions-positions[0],axis=1)>=reach)
        if not len(candidates):
            raise ValueError('prediction too short for moving connection')
        for index in candidates[::max(1,len(candidates)//8)]:
            advance=base_times[index];end,ve,ae=self._base(advance)
            distance=np.linalg.norm(end-positions[0])
            nominal=2*distance/max(np.linalg.norm(v)+np.linalg.norm(ve),.1)
            for duration in nominal*np.array([1.,1.3,1.7]):
                prefix=BPoly.from_derivatives([0.,duration],[[positions[0],v,a],[end,ve,ae]])
                ts=np.linspace(0.,duration,max(65,int(duration/.02)+1))
                xyz=prefix(ts)
                if self._limit_ratio(prefix(ts,nu=1),prefix(ts,nu=2),prefix(ts,nu=3),
                                     max_speed,max_acceleration,max_jerk,max_yaw_rate)>1.01:
                    continue
                pv,pa=prefix(ts,nu=1),prefix(ts,nu=2)
                yaw_rate=(pv[:,0]*pa[:,1]-pv[:,1]*pa[:,0])/np.maximum(np.sum(pv[:,:2]**2,axis=1),1e-8)
                if np.abs(np.gradient(yaw_rate,ts)).max()>max_yaw_acceleration*1.01:
                    continue
                if path_distance(xyz,self.original).max()>deviation or (check and not check(xyz)):
                    continue
                self.prefix,self.advance=prefix,advance
                return
        raise ValueError('no continuous moving connection')

    def _finish(self):
        if self.direct is not None:
            self.times=self.direct.x
        else:
            shift=(self.prefix.x[-1] if self.prefix is not None else 0.)-self.advance
            times=self.clock.x[self.clock.x>self.advance]+shift
            self.times=np.r_[0.,times]
            if self.prefix is not None:
                self.times=np.unique(np.r_[self.times,self.prefix.x[-1]])
        self.duration=float(self.times[-1]);self.durations=np.diff(self.times)
        self.points=self.sample(self.times)[0]

    def sample(self,t):
        ts=np.asarray(np.clip(t,0.,self.duration if hasattr(self,'duration') else self.times[-1]))
        if self.direct is not None:
            return tuple(self.direct(ts,nu=n) for n in range(3))
        if self.prefix is None:
            return self._base(ts)
        end=self.prefix.x[-1]
        result=self._base(np.maximum(0,ts-end)+self.advance)
        return tuple(np.where((ts<end)[...,None],self.prefix(np.clip(ts,0,end),nu=n),r) for n,r in enumerate(result))

    def hold_start(self,duration):
        if self.prefix is not None:
            raise ValueError('cannot hold a moving trajectory')
        p=self._base(0.)[0]
        self.prefix=BPoly.from_derivatives([0.,duration],[[p,np.zeros(3),np.zeros(3)]]*2)
        self.advance=0.;self._finish()

    def remaining(self,t,actual,horizon=None):
        start=np.clip(t,0.,self.duration);end=self.duration if horizon is None else min(self.duration,start+horizon)
        ts=np.linspace(start,end,max(2,int(np.ceil((end-start)/.03))+1))
        return np.vstack([actual,self.sample(ts)[0]])

    def save(self,path):
        arrays=dict(times=self.times,advance=self.advance)
        for name in ('geometry','clock','prefix','direct'):
            curve=getattr(self,name,None)
            if curve is not None:
                arrays[name+'_c']=curve.c;arrays[name+'_x']=curve.x
        np.savez_compressed(path,**arrays)

    @classmethod
    def load(cls,path):
        result=cls.__new__(cls)
        with np.load(path) as data:
            result.advance=float(data['advance'])
            for name in ('geometry','clock','prefix','direct'):
                setattr(result,name,BPoly(data[name+'_c'],data[name+'_x']) if name+'_c' in data else None)
        result._finish();return result

    @classmethod
    def brake(cls,p,v,a,max_acceleration=1.,max_jerk=4.):
        p,v,a=map(np.asarray,(p,v,a));result=cls.__new__(cls)
        result.prefix=None;result.advance=0.
        duration=max(.25,2*np.linalg.norm(v)/max_acceleration,np.sqrt(8*np.linalg.norm(v)/max_jerk),2*np.linalg.norm(a)/max_jerk)
        for _ in range(12):
            end=p+.5*duration*v+duration**2*a/12
            curve=BPoly.from_derivatives([0.,duration],[[p,v,a],[end,np.zeros(3),np.zeros(3)]])
            ts=np.linspace(0.,duration,81)
            if np.linalg.norm(curve(ts,nu=2),axis=1).max()<=max_acceleration*1.001 and np.linalg.norm(curve(ts,nu=3),axis=1).max()<=max_jerk*1.001:
                result.direct=curve;result._finish();return result
            duration*=1.2
        raise ValueError('cannot fit braking trajectory')

def connect_path(points, position, velocity, max_acceleration):
    """投影到原预测折线，只保留前方部分；坐标系由调用方固定在观测时刻。"""
    points, position = np.asarray(points), np.asarray(position)
    delta = np.diff(points, axis=0)
    alpha = np.clip(np.sum((position-points[:-1])*delta, axis=1)/
                    np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0., 1.)
    projections = points[:-1]+alpha[:, None]*delta
    i = np.argmin(np.linalg.norm(projections-position, axis=1))
    tail = points[i+1:]
    # 留出连续接入的距离，避免将仍在前进的速度塞进极短的首段。
    distance = max(.15, np.linalg.norm(velocity)**2/(2*max_acceleration))
    while len(tail) > 1 and np.linalg.norm(tail[0]-position) < distance:
        tail = tail[1:]
    if np.linalg.norm(tail[-1]-position) < .15:
        raise ValueError('prediction already traversed')
    return np.vstack([position, tail])


class Heading:
    def __init__(self, yaw, max_rate=1., max_acceleration=2.):
        self.yaw, self.rate = float(yaw), 0.
        self.max_rate, self.max_acceleration = max_rate, max_acceleration

    def target(self, path, t):
        p, v, _ = path.sample(t)
        direction = path.sample(min(t+.5, path.duration))[0]-p
        if np.linalg.norm(direction[:2]) < .02:
            direction = v
        if np.linalg.norm(direction[:2]) < .02:
            i = min(np.searchsorted(path.times, t, side='right'), len(path.points)-1)
            while i < len(path.points)-1 and np.linalg.norm(path.points[i]-p) < .02:
                i += 1
            direction = path.points[i]-p
        return self.yaw if np.linalg.norm(direction[:2]) < .02 else float(np.arctan2(direction[1], direction[0]))

    def step(self, target, dt):
        error = np.arctan2(np.sin(target-self.yaw), np.cos(target-self.yaw))
        rate = np.sign(error)*min(self.max_rate, np.sqrt(2*self.max_acceleration*abs(error)), abs(error)/dt)
        self.rate += np.clip(rate-self.rate, -self.max_acceleration*dt, self.max_acceleration*dt)
        self.yaw += self.rate*dt
        return self.yaw
