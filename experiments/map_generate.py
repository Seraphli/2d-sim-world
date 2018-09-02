import numpy as np


def create_mesh_grid(w, h):
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    mesh_grid = np.stack(np.meshgrid(x, y, 0), 2)
    mesh_grid = np.reshape(mesh_grid, (h, w, 3))
    mesh_grid = np.transpose(mesh_grid, (1, 0, 2))
    return mesh_grid


class Cloud(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.radius_limit = (50, 200)
        self.rain_drop_limit = (200, 500)
        self.mesh_grid = create_mesh_grid(w, h)
        self.state = 'Stop'

    def start_rain(self):
        self.center = [np.random.randint(0, self.w, 1)[0],
                       np.random.randint(0, self.h, 1)[0]]
        self.center = np.array(self.center)
        self.radius = np.random.randint(self.radius_limit[0],
                                        self.radius_limit[1] + 1, 1)[0]
        self.speed = [np.random.randint(-5, 5, 1)[0],
                      np.random.randint(-5, 5, 1)[0]]
        self.speed = np.array(self.speed)
        self.duration = np.random.randint(200, 500, 1)[0]
        self.i = 1
        self.rain_drop_num = np.random.randint(
            self.rain_drop_limit[0], self.rain_drop_limit[1] + 1, 1)[0]
        self.state = 'Raining'

    def rain_fall_step(self):
        if self.i >= self.duration:
            self.state = 'Stop'
            return
        rain_drops = [np.random.randint(0, self.radius, self.rain_drop_num),
                      np.random.randint(0, self.radius, self.rain_drop_num)]
        rain_drops = np.stack(rain_drops, 1)
        rain_drops = rain_drops + self.center
        indexes = np.where(np.logical_and(rain_drops[:, 0] < self.w,
                                          rain_drops[:, 1] < self.h))

        rain_drops = rain_drops[indexes]
        indexes = np.where(np.logical_and(rain_drops[:, 0] > 0,
                                          rain_drops[:, 1] > 0))
        rain_drops = rain_drops[indexes]
        v = np.zeros((self.w, self.h))
        v[rain_drops[:, 0], rain_drops[:, 1]] = 0.2
        if np.any(rain_drops):
            if self.i % 10 == 0:
                self.center += self.speed
            self.i += 1
        else:
            self.i = self.duration
        return v


class World(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.mesh_grid = create_mesh_grid(w, h)
        self.attr = {
            'z': np.zeros((w, h)),
            'humidity': np.zeros((w, h))
        }

    def init_height_map(self):
        z_lim = (-100, 100)
        point_num = 4
        m_lim = (20, 50)
        m_dis = m_lim[1] - m_lim[0]
        m = np.random.normal(m_dis / 2, m_dis / 6, point_num)
        m = np.abs(m - m_dis / 2) + m_lim[0]
        z_dis = z_lim[1] - z_lim[0]
        z_center = sum(z_lim) / 2
        z = np.random.normal(z_center, z_dis / 3, point_num)
        mass_point = [
            np.random.randint(0, self.w, point_num),
            np.random.randint(0, self.h, point_num), z]
        mass_point = np.stack(mass_point, 1)
        z = np.zeros((self.w, self.h))
        for n in range(point_num):
            r = np.sqrt(np.sum(np.power(mass_point[n] - self.mesh_grid, 2), 2))
            r_3 = np.repeat(np.reshape(r, (self.w, self.h, 1)), 3, 2)
            _v = np.divide(np.divide(np.divide(
                mass_point[n] - self.mesh_grid, r_3) * m[n], r_3), r_3)
            z += np.dot(_v, np.array([0, 0, 1]))
        self.attr['z'] = (z - np.min(z)) / (np.max(z) - np.min(z))

    @property
    def value(self):
        z = self.attr['humidity']
        color = (z * 0xff).astype(np.uint32)
        return color + color * 0x100 + color * 0x10000


class MapGenerate(object):
    def __init__(self, width=1280, height=720):
        self.w = width
        self.h = height
        self.world = World(width, height)
        self.t = 0

    def reset(self):
        self.world.init_height_map()
        self.cloud = Cloud(self.w, self.h)
        self.cloud.start_rain()

    def step(self):
        h = self.cloud.rain_fall_step()
        if h is not None:
            self.world.attr['humidity'] += h
            indexes = np.where(self.world.attr['humidity'] > 1)
            if len(indexes[0]) > 0:
                _index = []
                for i in range(len(indexes[0])):
                    _index.append([indexes[0][i], indexes[1][i]])
                indexes = _index
                shift = [[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0],
                         [1, 1], [1, -1], [-1, 1], [-1, -1]]
                while len(indexes) > 0:
                    print(len(indexes))
                    index = indexes[0]
                    print(index)
                    _index = [np.array(index) + np.array(s) for s in shift]
                    _index = [_i for _i in _index
                              if 0 <= _i[0] < self.w and 0 <= _i[1] < self.h]
                    z_list = [self.world.attr['z'][_i[0], _i[1]]
                              for _i in _index]
                    next_index = np.array(_index)[np.argmin(z_list)]
                    if np.all(next_index == index):
                        self.world.attr['humidity'][index[0], index[1]] = 1
                    else:
                        self.world.attr['humidity'][
                            next_index[0], next_index[1]] += \
                            self.world.attr['humidity'][index[0], index[1]] - 1
                        self.world.attr['humidity'][index[0], index[1]] = 1
                        if self.world.attr['humidity'][
                            next_index[0], next_index[1]] > 1:
                            indexes.append(next_index)
                    indexes = indexes[1:]

        self.world.attr['humidity'] -= np.ones((self.w, self.h)) * 0.0005
        self.world.attr['humidity'][self.world.attr['humidity'] < 0] = 0
        self.t += 1
        if self.t % 50 == 0 and self.cloud.state == 'Stop':
            self.cloud.start_rain()
        return self.world.value, {}


if __name__ == '__main__':
    from gwe.gwe import GridWorldEngine

    sim = MapGenerate()
    sim.reset()
    gwe = GridWorldEngine(sim_obj=sim)
    gwe.render(visible=True, speed=1)
    while True:
        gwe.step()
