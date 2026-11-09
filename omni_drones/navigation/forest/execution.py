"""Physics-clock reference playback with measured rest and heading gates."""
import numpy as np
from .geometry import yaw_from_quat


def wrap(angle):
    return (angle+np.pi) % (2*np.pi)-np.pi


def course_reference(path, t):
    p, v, a, j = path._base(t, True)
    speed2 = float(v[:2]@v[:2])
    tangent = v if speed2 > 1e-8 else path.geometry(float(path.clock(t)), nu=1)
    yaw = float(np.arctan2(tangent[1], tangent[0]))
    vv = max(speed2, 1e-8)
    rate = float((v[0]*a[1]-v[1]*a[0])/vv)
    acceleration = float((v[0]*j[1]-v[1]*j[0])/vv-2*rate*(v[:2]@a[:2])/vv)
    return p, v, a, yaw, rate, acceleration


class ReferencePlayer:
    def __init__(self, path, segments, phases, initial_yaw, settle_seconds=3.):
        self.path, self.segments, self.phases = path, segments, phases
        self.index = 0
        self.time = 0.
        self.yaw = initial_yaw
        self.settle_seconds = settle_seconds
        self.ready_seconds = self.wait_seconds = 0.
        self.complete = False
        self.failure = None
        self.events = []
        self.yaw_rate = self.yaw_acceleration = 0.

    def reference(self):
        segment = self.segments[self.index]
        p, v, a = self.path.sample(self.time)
        self.yaw_rate = self.yaw_acceleration = 0.
        if segment['kind'] == 'ROTATE':
            u = np.clip((self.time-segment['start_s'])/(segment['end_s']-segment['start_s']), 0., 1.)
            self.yaw = segment['yaw_start']+(segment['yaw_end']-segment['yaw_start'])*(10*u**3-15*u**4+6*u**5)
            delta = segment['yaw_end']-segment['yaw_start']; duration = segment['end_s']-segment['start_s']
            self.yaw_rate = delta*(30*u**2-60*u**3+30*u**4)/duration
            self.yaw_acceleration = delta*(60*u-180*u**2+120*u**3)/duration**2
        elif segment['kind'] == 'TRANSIT':
            self.yaw = segment['yaw_start']
        elif np.linalg.norm(v[:2]) > .01:
            self.yaw = float(np.arctan2(v[1], v[0]))
            *_, self.yaw_rate, self.yaw_acceleration = course_reference(self.path, self.time)
        elif self.time <= segment['start_s']+1e-8:
            self.yaw = segment['yaw_start']
        else:
            self.yaw = segment['yaw_end']
        return p, v, a, self.yaw

    def advance(self, dt, state, simulated_time):
        """Call only AFTER a successful physics step, with the measured new state."""
        if dt <= 0 or self.complete or self.failure: return
        segment = self.segments[self.index]
        initial = simulated_time < self.settle_seconds or (self.index == 0 and self.time == 0.)
        boundary = self.time >= segment['end_s']-1e-8
        if initial or boundary:
            p = self.path.sample(self.time)[0]
            yaw = segment['yaw_start'] if initial else segment['yaw_end']
            ready = (np.linalg.norm(state[:3]-p) < .10 and np.linalg.norm(state[7:10]) < .10
                     and abs(wrap(yaw_from_quat(state[3:7])-yaw)) < np.deg2rad(5)
                     and np.linalg.norm(state[10:13]) < .12)
            self.wait_seconds += dt
            self.ready_seconds = self.ready_seconds+dt if ready else 0.
            if self.wait_seconds > 20.:
                self.failure = 'rest_gate_timeout'; return
            if self.ready_seconds < .3 or simulated_time < self.settle_seconds: return
            self.events.append(dict(segment=self.index, simulated_time=simulated_time,
                                    reference_time=self.time, wait_seconds=self.wait_seconds,
                                    speed=float(np.linalg.norm(state[7:10])),
                                    yaw_error_deg=float(np.rad2deg(wrap(yaw_from_quat(state[3:7])-yaw)))))
            self.ready_seconds = self.wait_seconds = 0.
            if boundary:
                if self.index == len(self.segments)-1:
                    self.complete = True; return
                self.index += 1
                segment = self.segments[self.index]
        self.time = min(self.time+dt, segment['end_s'])

    def metadata(self):
        phase = next((p['phase'] for p in self.phases if p['start_s'] <= self.time < p['end_s']), 'EXIT')
        waiting = (self.time >= self.segments[self.index]['end_s']-1e-8 or self.wait_seconds > 0
                   or (self.index == 0 and self.time == 0.))
        return dict(reference_time=self.time, phase=phase, execution_segment=self.index,
                    execution_state='WAIT' if waiting else self.segments[self.index]['kind'],
                    default_navigation_training=phase == 'INSPECT' and not waiting and not self.complete)


class ContinuousReferencePlayer(ReferencePlayer):
    """Continuous curve timing, with measured start/end gates and tangent heading."""
    def __init__(self, path, phases, initial_yaw, settle_seconds=3.):
        final_yaw = course_reference(path, path.duration)[3]
        super().__init__(path, [dict(kind='CRUISE', start_s=0., end_s=path.duration,
                                    yaw_start=initial_yaw, yaw_end=final_yaw)], phases, initial_yaw, settle_seconds)

    def reference(self):
        p, v, a, self.yaw, self.yaw_rate, self.yaw_acceleration = course_reference(self.path, self.time)
        return p, v, a, self.yaw

    def metadata(self):
        result = super().metadata()
        if result['execution_state'] != 'WAIT' and result['phase'] == 'TURN':
            result['execution_state'] = 'U_TURN'
        return result
