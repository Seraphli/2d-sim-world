import numpy as np


class World(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.v = np.zeros((w, h))
        self.z = np.zeros((w, h))

    def init_height_map(self):
        z_lim = 100
        point_num = 4
        m = (5, 50)
        m_range = m[1] - m[0]
        m = np.random.normal(m_range / 2, m_range / 6, point_num)
        m = np.abs(m - m_range / 2) + m[0]
        mass_point = [
            np.random.randint(0, self.x + 1, point_num),
            np.random.randint(0, self.y + 1, point_num),
            np.random.normal(0, z_lim / 3, point_num)]
        mass_point = np.stack(mass_point, 1)
        world = []
        for _x in range(self.x):
            for _y in range(self.y):
                world.append([_x, _y, 0])
        world = np.reshape(np.array(world), (self.x, self.y, 3))
        z = np.zeros((self.x, self.y))
        for n in range(point_num):
            r = np.sqrt(np.sum(np.power(mass_point[n] - world, 2), 2))
            r_3 = np.repeat(np.reshape(r, (self.x, self.y, 1)), 3, 2)
            _v = np.divide(np.divide(np.divide(
                mass_point[n] - world, r_3) * self.G * m[n], r_3), r_3)
            z += np.dot(_v, np.array([0, 0, 1]))
        self.world = z
        self.world = (self.world - np.min(self.world)) / \
                     (np.max(self.world) - np.min(self.world))
        self.world = (self.world * 0xff).astype(np.uint32)


class Cloud(object):
    def __init__(self, w, h):
        self.x = w
        self.y = h
        self.center = [np.random.randint(0, w + 1),
                       np.random.randint(0, h + 1)]
        self.speed = [np.random.randint(-10, 10),
                      np.random.randint(-10, 10)]
        self.duration = np.random.randint(0, 10)


class MapGenerate(object):
    def __init__(self, width=1280, height=720):
        self.x = width
        self.y = height
        self.z = 100
        self.m = (5, 50)
        self.point_num = 3
        self.G = 9.8

    def init_height_map(self):
        m_range = self.m[1] - self.m[0]
        m = np.random.normal(m_range / 2, m_range / 6, self.point_num)
        m = np.abs(m - m_range / 2) + self.m[0]
        mass_point = [
            np.random.randint(0, self.x + 1, self.point_num),
            np.random.randint(0, self.y + 1, self.point_num),
            np.random.normal(0, self.z / 3, self.point_num)]
        mass_point = np.stack(mass_point, 1)
        world = []
        for _x in range(self.x):
            for _y in range(self.y):
                world.append([_x, _y, 0])
        world = np.reshape(np.array(world), (self.x, self.y, 3))
        z = np.zeros((self.x, self.y))
        for n in range(self.point_num):
            r = np.sqrt(np.sum(np.power(mass_point[n] - world, 2), 2))
            r_3 = np.repeat(np.reshape(r, (self.x, self.y, 1)), 3, 2)
            _v = np.divide(np.divide(np.divide(
                mass_point[n] - world, r_3) * self.G * m[n], r_3), r_3)
            z += np.dot(_v, np.array([0, 0, 1]))
        self.world = z
        self.world = (self.world - np.min(self.world)) / \
                     (np.max(self.world) - np.min(self.world))
        self.world = (self.world * 0xff).astype(np.uint32)

    def reset(self):
        self.init_height_map()

    def step(self):
        return self.world, {}


if __name__ == '__main__':
    from gwe.gwe import GridWorldEngine

    sim = MapGenerate()
    sim.reset()
    gwe = GridWorldEngine(sim_obj=sim)
    gwe.render(visible=True, speed=1)
    while True:
        gwe.step()
