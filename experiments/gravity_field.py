import numpy as np
import multiprocessing as mp

G = 9.8


def func(args):
    _x, y, point_num, mass_point = args
    z = np.array([0, 0, 1])
    vals = []
    for _y in range(y):
        _val = 0
        for _i in range(point_num):
            v = mass_point[_i, :3] - np.array([_x, _y, 0])
            r = np.linalg.norm(v)
            if r > 0:
                v = v / (np.power(r, 3)) * G * mass_point[_i, 3]
                val = np.dot(v, z)
                _val += val
            else:
                _val += mass_point[_i, 2]
        vals.append(_val)
    return vals


class GravityField(object):
    def __init__(self, width=1280, height=720):
        self.x = width
        self.y = height
        self.z = 100
        self.m = (5, 50)
        self.point_num = 30
        self.pool = mp.Pool(mp.cpu_count() - 2)

    def reset(self):
        z = np.random.normal(0, self.z / 3, self.point_num)
        m_range = self.m[1] - self.m[0]
        m = np.random.normal(m_range / 2, m_range / 6, self.point_num)
        m = np.abs(m - m_range / 2) + self.m[0]
        self.mass_point = [
            np.random.randint(0, self.x + 1, self.point_num),
            np.random.randint(0, self.y + 1, self.point_num),
            z, m]
        self.mass_point = np.stack(self.mass_point, 1)
        self.world = np.zeros((self.x, self.y))
        results = self.pool.map(func, [
            (i, self.y, self.point_num, self.mass_point)
            for i in range(self.x)])
        self.world = np.array(results)
        self.world = (self.world - np.min(self.world)) / \
                     (np.max(self.world) - np.min(self.world))
        self.world = (self.world * 0xffffff).astype(np.uint32)

    def step(self):
        return self.world, {}


if __name__ == '__main__':
    from gwe.gwe import GridWorldEngine

    sim = GravityField()
    sim.reset()
    gwe = GridWorldEngine(sim_obj=sim)
    gwe.render(visible=True, speed=1)
    while True:
        gwe.step()
