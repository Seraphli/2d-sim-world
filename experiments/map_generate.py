import numpy as np
import pickle

EPSILON = 0.05


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
        self.center = [np.random.randint(50, self.w - 50, 1)[0],
                       np.random.randint(50, self.h - 50, 1)[0]]
        self.center = np.array(self.center)
        self.radius = np.random.randint(self.radius_limit[0],
                                        self.radius_limit[1] + 1, 1)[0]
        self.speed = [np.random.randint(-5, 5, 1)[0],
                      np.random.randint(-5, 5, 1)[0]]
        self.speed = np.array(self.speed)
        self.duration = np.random.randint(400, 2000, 1)[0]
        self.i = 1
        self.rain_drop_num = np.random.randint(
            self.rain_drop_limit[0], self.rain_drop_limit[1] + 1, 1)[0]
        self.state = 'Raining'
        print(self.state)

    def rain_fall_step(self):
        if self.i >= self.duration:
            self.state = 'Stop'
            return
        rain_drops = [np.random.randint(0, self.radius, self.rain_drop_num),
                      np.random.randint(0, self.radius, self.rain_drop_num)]
        rain_drops = np.stack(rain_drops, 1)
        rain_drops = rain_drops + self.center
        indexes = np.where(np.logical_and(rain_drops[:, 0] < self.w - 50,
                                          rain_drops[:, 1] < self.h - 50))

        rain_drops = rain_drops[indexes]
        indexes = np.where(np.logical_and(rain_drops[:, 0] > 50,
                                          rain_drops[:, 1] > 50))
        rain_drops = rain_drops[indexes]
        v = np.zeros((self.w, self.h))
        v[rain_drops[:, 0], rain_drops[:, 1]] = 0.3
        if np.any(rain_drops):
            if self.i % 10 == 0:
                self.center += self.speed
            self.i += 1
        else:
            self.i = self.duration
        return v


class HeightMap(object):
    def __init__(self, width=400, height=400):
        self.w = width
        self.h = height
        self.mesh_grid = create_mesh_grid(self.w, self.h)
        self.value = None

    def create(self):
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
        self.value = (z - np.min(z)) / (np.max(z) - np.min(z))

    def save(self):
        with open('height_map.pkl', 'wb') as f:
            pickle.dump(self.value, f)

    def load(self):
        with open('height_map.pkl', 'rb') as f:
            self.value = pickle.load(f)


class World(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.mesh_grid = create_mesh_grid(w, h)
        self.attr = {
            'height': HeightMap(),
            'water_level': np.zeros((w, h)),
            'humidity': np.zeros((w, h))
        }

    def create(self):
        self.attr['height'].create()
        self.attr['height'].save()

    def save(self):
        with open('world.pkl', 'wb') as f:
            pickle.dump(self.attr, f)

    def load(self):
        with open('world.pkl', 'rb') as f:
            self.attr = pickle.load(f)

    @property
    def value(self):
        # Display water level
        v = np.copy(self.attr['water_level'])
        v[v > 1] = 1

        # Display height map
        # v = self.attr['height'].value / np.max(self.attr['height'].value)

        # Display humidity
        # v = self.attr['humidity']
        color = (v * 0xff).astype(np.uint32)
        return color + color * 0x100 + color * 0x10000


class Weather(object):
    def __init__(self, world):
        self.world = world
        self.cloud = Cloud(self.world.width, self.world.height)
        self.cloud.start_rain()
        self.t = 0

    def cal_cond(self, z):
        shift = []
        blocks = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                shift.append([i, j])
                block = z[1 + i:-1 + i if -1 + i < 0 else None,
                        1 + j:-1 + j if -1 + j < 0 else None]
                blocks.append(block)

        blocks = np.stack(blocks, 2)
        center = z[1:-1, 1:-1]
        centers = np.repeat(np.reshape(center, (self.world.width - 2,
                                                self.world.height - 2, 1)), 8,
                            2)
        diff = centers - blocks
        water = self.world.attr['humidity'] + \
                self.world.attr['water_level']
        wl_cond = water[1:-1, 1:-1] >= EPSILON
        wl_cond = np.repeat(
            np.reshape(wl_cond,
                       (self.world.width - 2, self.world.height - 2, 1)),
            8, 2)
        cond = np.logical_and(diff > EPSILON * 2, wl_cond)
        return cond, diff, shift

    def step(self):
        # 关于湿度,如果水分大于周边的几倍就可以开始扩散
        # Rain drop
        h = self.cloud.rain_fall_step()
        if h is not None:
            # Water level will rise if humidity > 1
            water = self.world.attr['humidity'] + \
                    self.world.attr['water_level'] + h
            overflow = water - 1
            overflow[overflow < 0] = 0
            self.world.attr['water_level'] = overflow
            self.world.attr['humidity'] = water - overflow

        z = self.world.attr['height'].value + self.world.attr['humidity'] \
            + self.world.attr['water_level']
        # Find the gradient
        cond, diff, shift = self.cal_cond(z)
        if np.any(cond):
            indexes = np.argwhere(cond)
            point_dict = {}
            # print('l1: {}'.format(len(indexes)))
            for index in indexes:
                if (index[0], index[1]) in point_dict:
                    point_dict[(index[0], index[1])]. \
                        append((index[2],
                                diff[index[0], index[1], index[2]]))
                else:
                    point_dict[(index[0], index[1])] = \
                        [(index[2],
                          diff[index[0], index[1], index[2]])]

            indexes = []
            for k, v in point_dict.items():
                v = sorted(v, key=lambda x: x[1], reverse=True)
                indexes.append((k[0], k[1], v[0][0]))

            # Move water according to water direction
            # print('l2: {}'.format(len(indexes)))
            for index in indexes:
                # print(index)
                i = np.array([index[0], index[1]])
                real_i = i + np.array([1, 1])
                s = np.array(shift[index[2]])
                real_next_i = real_i + s
                d = diff[index[0], index[1], index[2]]
                if d > EPSILON * 2 * 2:
                    c = d / 4
                else:
                    c = EPSILON
                self.world.attr['humidity'][real_i[0],
                                            real_i[1]] -= c
                self.world.attr['humidity'][real_next_i[0],
                                            real_next_i[1]] += c
                if self.world.attr['humidity'][real_i[0],
                                               real_i[1]] < 0 or \
                        self.world.attr['humidity'][real_next_i[0],
                                                    real_next_i[1]] < 0:
                    print('error')
            # print('loop')

        # Calculate evaporation
        water = self.world.attr['humidity'] + \
                self.world.attr['water_level']
        water -= np.ones((self.world.width, self.world.height)) * 0.001
        water[water < 0] = 0
        overflow = water - 1
        overflow[overflow < 0] = 0
        self.world.attr['water_level'] = overflow
        self.world.attr['humidity'] = water - overflow
        self.t += 1
        if self.t % 50 == 0 and self.cloud.state == 'Stop':
            self.cloud.start_rain()


class WorldSimulator(object):
    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        self.world = World(width, height)
        self.weather = Weather(self.world)

    def reset(self):
        # self.world.create()
        # self.world.save()
        self.world.load()

    def step(self):
        self.weather.step()
        return self.world.value, {}


if __name__ == '__main__':
    from gwe.gwe import GridWorldEngine

    sim = WorldSimulator()
    sim.reset()
    gwe = GridWorldEngine(sim_obj=sim)
    gwe.render(visible=True, speed=1)
    while True:
        gwe.step()
