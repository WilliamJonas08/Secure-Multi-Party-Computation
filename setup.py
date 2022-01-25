# coding:utf-8

import time
import tkinter
import io
import random
import config as cfg
import qlearn
from queue import Queue


class Cell:
    def __init__(self):
        self.wall = False

    def color(self):
        if self.wall:
            return cfg.wall_color
        else:
            return 'white'

    def load(self, data):
        if data == 'X':
            self.wall = True
        else:
            self.wall = False

#TODO : pas trouvé ou c'est utilisé ça
    def __getattr__(self, key):
        if key == 'neighbors':
            opts = [self.world.get_next_grid(self.x, self.y, dir) for dir in range(self.world.directions)]
            next_states = tuple(self.world.grid[y][x] for (x, y) in opts)
            return next_states
        raise AttributeError(key)

class Agent:
    def __setattr__(self, key, value):
        if key == 'cell':
            old = self.__dict__.get(key, None)
            if old is not None:
                old.agents.remove(self)
            if value is not None:
                value.agents.append(self)
        self.__dict__[key] = value

    def go_direction(self, dir):
        target = self.cell.neighbors[dir]
        if getattr(target, 'wall', False):
            print("hit a wall")
            return False
        self.cell = target
        return True

#
# class World:
#     def __init__(self, cell=None, directions=cfg.directions, filename=None):
#         if cell is None:
#             cell = Cell
#         self.Cell = cell
#         self.display = make_display(self)
#         self.directions = directions
#         self.filename = filename
#
#         self.grid = None
#         self.dictBackup = None
#         self.agents = []
#         self.age = 0
#
#         self.height = None
#         self.width = None
#         self.get_file_size(filename)
#
#         self.image = None
#         self.mouseWin = 0   #None
#         self.catWin = 0     #None
#         self.reset()
#         self.load(filename)
#
#     def get_file_size(self, filename):
#         if filename is None:
#             raise Exception("world file not exist!")
#         with open(filename) as f:
#             data = f.readlines()
#         if self.height is None:
#             self.height = len(data)
#         if self.width is None:
#             self.width = max([len(x.rstrip()) for x in data])
#
#     def reset(self):
#         self.grid = [[self.make_cell(i, j) for i in range(self.width)] for j in range(self.height)]
#         self.dictBackup = [[{} for _i in range(self.width)] for _j in range(self.height)]
#         self.agents = []
#         self.age = 0
#
#     def make_cell(self, x, y):
# #TODO : pas compris self.Cell callable ...
#         c = self.Cell()
#         c.x = x
#         c.y = y
# #TODO : c.world et c.agents definis ou ??? (class Cell tjr ?)
#         c.world = self
#         c.agents = []
#         return c
#
#     def get_cell(self, x, y):
#         return self.grid[y][x]
#
#     def get_relative_cell(self, x, y):
#         return self.grid[y % self.height][x % self.width]
#
#     def load(self, f):
#         if not hasattr(self.Cell, 'load'):
#             return
#         if isinstance(f, type('')):
#             with open(f) as f:
#                 lines = f.readlines()
#         else:
#             lines = f.readlines()
#         lines = [x.rstrip() for x in lines]
#         fh = len(lines)
#         fw = max([len(x) for x in lines])
#
#         if fh > self.height:
#             fh = self.height
#             start_y = 0
#         else:
#             start_y = (self.height - fh) // 2
#         if fw > self.width:
#             fw = self.width
#             start_x = 0
#         else:
#             start_x = (self.width - fw) // 2
#
#         self.reset()
#         for j in range(fh):
#             line = lines[j]
#             for i in range(min(fw, len(line))):
#                 self.grid[start_y + j][start_x + i].load(line[i])
#
#     def update(self, mouse_win=None, cat_win=None):
#         if hasattr(self.Cell, 'update'):
#             for a in self.agents:
#                 a.update()
#             self.display.redraw()
#         else:
#             for a in self.agents:
#                 old_cell = a.cell
#                 a.update()
#                 if old_cell != a.cell:  # old cell won't disappear when new cell
#                     self.display.redraw_cell(old_cell.x, old_cell.y)
#
#                 self.display.redraw_cell(a.cell.x, a.cell.y)
#
#         if mouse_win:
#             self.mouseWin = mouse_win
#         if cat_win:
#             self.catWin = cat_win
#         self.display.update()
#         self.age += 1
#
#     def get_next_grid(self, x, y, dir):
#         dx = 0
#         dy = 0
#         if self.directions == 8:
#             dx, dy = [(0, -1), (1, -1), (
#                 1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)][dir]
#         elif self.directions == 4:
#             dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][dir]
#         elif self.directions == 6:
#             if y % 2 == 0:
#                 dx, dy = [(1, 0), (0, 1), (-1, 1), (-1, 0),
#                           (-1, -1), (0, -1)][dir]
#             else:
#                 dx, dy = [(1, 0), (1, 1), (0, 1), (-1, 0),
#                           (0, -1), (1, -1)][dir]
#
#         x2 = x + dx
#         y2 = y + dy
#
#         if x2 < 0:
#             x2 += self.width
#         if y2 < 0:
#             y2 += self.height
#         if x2 >= self.width:
#             x2 -= self.width
#         if y2 >= self.height:
#             y2 -= self.height
#
#         return x2, y2
#
#     def add_agent(self, agent, x=None, y=None, cell=None, dir=None):
#         self.agents.append(agent)
#         if cell is not None:
#             x = cell.x
#             y = cell.y
#         if x is None:
#             x = random.randrange(self.width)
#         if y is None:
#             y = random.randrange(self.height)
#         if dir is None:
#             dir = random.randrange(self.directions)
#
#         agent.cell = self.grid[y][x]
#         agent.dir = dir
#         agent.world = self


class World:
    def __init__(self, cell=None, directions=cfg.directions, filename=None):
        self.cat = Cat(filename=filename)
        self.mouse = Mouse()
        if cell is None:
            cell = Cell
        self.Cell = cell
        self.display = make_display(self)
        self.directions = directions
        self.filename = filename

        self.grid = None
        self.dictBackup = None
        self.agents = [self.cat, self.mouse]
        self.age = 0

        self.height = None
        self.width = None
        self.get_file_size(filename)

        self.image = None
        self.mouseWin = 0
        self.catWin = 0
        self.reset()
        self.load(filename)

    def get_file_size(self, filename):
        if filename is None:
            raise Exception("world file not exist!")
        with open(filename) as f:
            data = f.readlines()
        if self.height is None:
            self.height = len(data)
        if self.width is None:
            self.width = len(data[0])-1

    def reset(self):
        self.grid = [[self.make_cell(i, j) for i in range(self.width)] for j in range(self.height)]
        self.dictBackup = [[{} for _i in range(self.width)] for _j in range(self.height)]
        self.agents = []
        self.age = 0

    def make_cell(self, x, y):
        c = self.Cell()
        c.x = x
        c.y = y
        c.world = self
        c.agents = []
        return c

    def get_cell(self, x, y):
        return self.grid[y][x]

    def get_relative_cell(self, x, y):
        return self.grid[y % self.height][x % self.width]

    def load(self, f):
        if not hasattr(self.Cell, 'load'):
            return
        if isinstance(f, type('')):
            with open(f) as f:
                lines = f.readlines()
        else:
            lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        fh = len(lines)
        fw = max([len(x) for x in lines])

        if fh > self.height:
            fh = self.height
            start_y = 0
        else:
            start_y = (self.height - fh) // 2
        if fw > self.width:
            fw = self.width
            start_x = 0
        else:
            start_x = (self.width - fw) // 2

        self.reset()
        for j in range(fh):
            line = lines[j]
            for i in range(min(fw, len(line))):
                self.grid[start_y + j][start_x + i].load(line[i])

    def update(self, mouse_win=None, cat_win=None):
        if hasattr(self.Cell, 'update'):
            self.update_mouse()
            self.update_cat()
            self.display.redraw()
        else:
            old_cell_cat = self.cat.cell
            old_cell_mouse = self.mouse.cell
            self.update_cat()
            self.update_mouse()
            if old_cell_cat != self.cat.cell:
                self.display.redraw_cell(old_cell_cat.x, old_cell_cat.y)
            if old_cell_mouse != self.mouse.cell:
                self.display.redraw_cell(old_cell_mouse.x, old_cell_mouse.y)

            self.display.redraw_cell(self.cat.cell.x, self.cat.cell.y)
            self.display.redraw_cell(self.mouse.cell.x, self.mouse.cell.y)
        if mouse_win:
            self.mouseWin = mouse_win
        if cat_win:
            self.catWin = cat_win
        self.display.update()
        self.age += 1

    def get_next_grid(self, x, y, dir):
        dx = 0
        dy = 0
        if self.directions == 8:
            dx, dy = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)][dir]
        elif self.directions == 4:
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][dir]

        x2 = x + dx
        y2 = y + dy

        if x2 < 0:
            x2 += self.width
        if y2 < 0:
            y2 += self.height
        if x2 >= self.width:
            x2 -= self.width
        if y2 >= self.height:
            y2 -= self.height

        return x2, y2

    def add_agents(self):
        self.mouse.cell = self.pick_random_location()  #self.grid[random.randrange(self.height)][random.randrange(self.width)]
        self.mouse.dir = random.randrange(self.directions)

        self.cat.cell = self.pick_random_location()
        self.cat.dir = random.randrange(self.directions)

    #TODO : move method into Mouse class
    def update_mouse(self):
        print('mouse update begin...')
        state = self.calculate_state_mouse()
        reward = cfg.MOVE_REWARD
        self.mouse.current_life_duration+=1
        if self.mouse.cell == self.cat.cell:
            print('eaten by cat...')
            self.catWin += 1
            reward = cfg.EATEN_BY_CAT
            if self.mouse.lastState is not None:
                self.mouse.ai.learn(self.mouse.lastState, self.mouse.lastAction, state, reward, is_last_state=True)
                print('mouse learn...')
            self.mouse.lastState = None
            self.mouse.life_durations.append(self.mouse.current_life_duration)
            self.mouse.current_life_duration = 0
            self.mouse.cell = self.pick_random_location()
            print('mouse random generate..')
            return

        elif self.mouse.current_life_duration >= cfg.TIME_TO_SURVIVE: #On définit le mouseWin
            self.mouseWin += 1
            self.lastState = None
            #TODO : apprendre aussi ici
            self.mouse.life_durations.append(self.mouse.current_life_duration)
            self.mouse.current_life_duration = 0
            self.cell = self.pick_random_location()
            print('mouse random generate..')
            return

        if self.mouse.lastState is not None: #souris non mangée
            self.mouse.ai.learn(self.mouse.lastState, self.mouse.lastAction, state, reward, is_last_state=False)

        # choose a new action and execute it
        action = self.mouse.ai.choose_action(state)
        self.mouse.lastState = state
        self.mouse.lastAction = action
        self.mouse.go_direction(action)

    #TODO : move method into Cat class
    def update_cat(self):
        print('cat update begin..')
        if self.cat.cell != self.mouse.cell:
            self.bfs_move(self.mouse.cell)
            print('cat move..')

    def calculate_state_mouse(self):
        def cell_value(cell):
            if self.cat.cell is not None and (cell.x == self.cat.cell.x and cell.y == self.cat.cell.y):
                return 3
            else:
                return 0 if cell.wall else 0

        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return tuple([cell_value(self.get_relative_cell(self.mouse.cell.x + dir[0], self.mouse.cell.y + dir[1])) for dir in dirs])

    def get_value(self, mdict, key):
        try:
            return mdict[key]
        except KeyError:
            return 0

    def pick_random_location(self):
        while 1:
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            cell = self.get_cell(x, y)
            if not (cell.wall or len(cell.agents) > 0):
                return cell

        # using BFS algorithm to move quickly to target.

    #TODO : move method into Cat class
    def bfs_move(self, target):
        if self.cat.cell == target:
            return

        for n in self.cat.cell.neighbors:
            if n == target:
                self.cat.cell = target  # if next move can go towards target
                return

        best_move = None
        q = Queue()
        start = (self.cat.cell.y, self.cat.cell.x)
        end = (target.y, target.x)
        q.put(start)
        step = 1
        V = {}
        preV = {}
        V[(start[0], start[1])] = 1

        print('begin BFS......')
        while not q.empty():
            grid = q.get()

            for i in range(8):
                ny, nx = grid[0] + self.cat.move[i][0], grid[1] + self.cat.move[i][1]
                if nx < 0 or ny < 0 or nx > (self.cat.fw - 1) or ny > (self.cat.fh - 1):
                    continue
                if self.get_value(V, (ny, nx)) or self.cat.grid_list[ny][nx] == 1:  # has visit or is wall.
                    continue

                preV[(ny, nx)] = self.get_value(V, (grid[0], grid[1]))
                if ny == end[0] and nx == end[1]:
                    V[(ny, nx)] = step + 1
                    seq = []
                    last = V[(ny, nx)]
                    while last > 1:
                        k = [key for key in V if V[key] == last]
                        seq.append(k[0])
                        assert len(k) == 1
                        last = preV[(k[0][0], k[0][1])]
                    seq.reverse()
                    print(seq)

                    best_move = self.grid[seq[0][0]][seq[0][1]]

                q.put((ny, nx))
                step += 1
                V[(ny, nx)] = step

        if best_move is not None:
            self.cat.cell = best_move

        else:
            dir = random.randrange(cfg.directions)
            self.cat.go_direction(dir)
            print("!!!!!!!!!!!!!!!!!!")

    def get_mouse_performance(self):
        return(self.mouse.life_durations)


class Mouse(Agent):
    def __init__(self):
        self.ai = None
        if cfg.LEARNING_MODE == 'Tabular Q-Learning':
            self.ai = qlearn.QLearn_Tabular(actions=range(cfg.directions), alpha=0.1, gamma=0.9, epsilon=0.1)
        elif cfg.LEARNING_MODE == 'Deep Q-Learning':
            self.ai = qlearn.QLearn(actions=range(cfg.directions), input_size=8, alpha=0.1, gamma=0.9, epsilon=0.1)
        self.catWin = 0
        self.mouseWin = 0

        self.lastState = None
        self.lastAction = None
        self.color = cfg.mouse_color

        self.current_life_duration=0
        self.life_durations=[]

        print('mouse init...')


class Cat(Agent):
    def __init__(self, filename):
        self.cell = None
        self.catWin = 0
        self.color = cfg.cat_color
        with open(filename) as f:
            lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        self.fh = len(lines) #height plate
        self.fw = max([len(x) for x in lines]) #width plate
        self.grid_list = [[1 for x in range(self.fw)] for y in range(self.fh)]
        self.move = [(0, -1), (1, -1), (
            1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

        for y in range(self.fh):
            line = lines[y]
            for x in range(min(self.fw, len(line))):
                t = 1 if (line[x] == 'X') else 0
                self.grid_list[y][x] = t

        print('cat init success......')

# GUI display
class tkinterDisplay:
    def __init__(self, size=cfg.grid_width):
        self.activated = False
        self.paused = False
        self.title = ''
        self.updateEvery = 1
        self.root = None
        self.speed = cfg.speed
        self.bg = None
        self.size = size
        self.imageLabel = None
        self.frameWidth = 0
        self.frameHeight = 0
        self.world = None
        self.bg = None
        self.image = None

    def activate(self):
        if self.root is None:
            self.root = tkinter.Tk()
        for c in self.root.winfo_children():
            c.destroy()
        self.bg = None
        self.activated = True
        self.imageLabel = tkinter.Label(self.root)
        self.imageLabel.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)
        self.frameWidth, self.frameHeight = self.world.width * self.size, self.world.height * self.size
        self.root.geometry('%dx%d' % (self.world.width * self.size, self.world.height * self.size))
        self.root.update()
        self.redraw()
        self.root.bind('<Escape>', self.quit)

    def quit(self):
        self.root.destroy()

    def update(self):
        if not self.activated:
            return
        if self.world.age % self.updateEvery != 0 and not self.paused:
            return
        self.set_title(self.title)
        self.imageLabel.update()
        if self.speed > 0:
            time.sleep(float(1)/self.speed)

    def make_title(self, world):
        text = 'age: %d' % world.age
        extra = []
        #if world.mouseWin:
        extra.append('mouseWin=%d' % world.mouseWin)
        #if world.catWin:
        extra.append('catWin=%d' % world.catWin)
        if world.display.paused:
            extra.append('paused')
        if world.display.updateEvery != 1:
            extra.append('skip=%d' % world.display.updateEvery)
        if world.display.speed > 0:
            extra.append('speed=%dm/s' % world.display.speed)

        if len(extra) > 0:
            text += ' [%s]' % ', '.join(extra)
        return text

    def set_title(self, title):
        if not self.activated:
            return
        self.title = title
        title += ' %s' % self.make_title(self.world)
        if self.root.title() != title:
            self.root.title(title)

    def pause(self, event=None):
        self.paused = not self.paused
        while self.paused:
            self.update()

    def getBackground(self):
        if self.bg is None:
            r, g, b = self.imageLabel.winfo_rgb(self.root['background'])
            self.bg = b'%c%c%c' % (r >> 8, g >> 8, b >> 8)
        return self.bg

    def redraw(self):
        if not self.activated:
            return

        iw = self.world.width * self.size
        ih = self.world.height * self.size

        hexgrid = self.world.directions == 6
        if hexgrid:
            iw += self.size / 2

        with open('temp.ppm', 'wb') as f:
            f.write(b'P6\n%d %d\n255\n' % (iw, ih))

            odd = False
            for row in self.world.grid:
                line = io.StringIO()
                if hexgrid and odd:
                    line.write(self.getBackground() * (self.size / 2))
                for cell in row:
                    if len(cell.agents) > 0:
                        c = self.get_data_color(cell.agents[-1])
                    else:
                        c = self.get_data_color(cell)

                    line.write(c * self.size)
                if hexgrid and not odd:
                    line.write(self.getBackground() * (self.size / 2))
                odd = not odd

                f.write((line.getvalue() * self.size).encode('latin1'))
        # f.close()

        self.image = tkinter.PhotoImage(file='temp.ppm')
        self.imageLabel.config(image=self.image)

    imageCache = {}

    def redraw_cell(self, x, y):
        if not self.activated:
            return
        sx = x * self.size
        sy = y * self.size
        if y % 2 == 1 and self.world.directions == 6:
            sx += self.size / 2

        cell = self.world.grid[y][x]
        if len(cell.agents) > 0:
            c = self.get_text_color(cell.agents[-1])
        else:
            c = self.get_text_color(cell)

        sub = self.imageCache.get(c, None)
        if sub is None:
            sub = tkinter.PhotoImage(width=1, height=1)
            sub.put(c, to=(0, 0))
            sub = sub.zoom(self.size)
            self.imageCache[c] = sub
        self.image.tk.call(self.image, 'copy', sub, '-from', 0, 0, self.size, self.size, '-to', sx, sy)

    def get_text_color(self, obj):
        c = getattr(obj, 'color', None)
        if c is None:
            c = getattr(obj, 'color', 'white')
        if callable(c):
            c = c()
        if isinstance(c, type(())):
            if isinstance(c[0], type(0.0)):
                c = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            return '#%02x%02x%02x' % c
        return c

    dataCache = {}

    def get_data_color(self, obj):
        c = getattr(obj, 'color', None)
        if c is None:
            c = getattr(obj, 'color', 'white')
        if callable(c):
            c = c()
        if isinstance(c, type(())):
            if isinstance(c[0], type(0.0)):
                c = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            return '%c%c%c' % c
        else:
            val = self.dataCache.get(c, None)
            if val is None:
                r, g, b = self.imageLabel.winfo_rgb(c)
                val = '%c%c%c' % (r >> 8, g >> 8, b >> 8)
                self.dataCache[c] = val
            return val

def make_display(world):
    d = tkinterDisplay()
    d.world = world
    return d


####Second World

class World_bis:
    def __init__(self, cell=None, directions=cfg.directions, filename=None):
        if cell is None:
            cell = Cell
        self.Cell = cell
        self.display = make_display_bis(self)
        self.directions = directions
        self.filename = filename

        self.grid = None
        self.dictBackup = None
        self.agents = []
        self.age = 0

        self.height = None
        self.width = None
        self.get_file_size(filename)

        self.image = None
        self.mouseWin = 0#None
        self.catWin = None
        self.reset()
        self.load(filename)

    def get_file_size(self, filename):
        if filename is None:
            raise Exception("world file not exist!")
        with open(filename) as f:
            data = f.readlines()
        if self.height is None:
            self.height = len(data)
        if self.width is None:
            self.width = max([len(x.rstrip()) for x in data])

    def reset(self):
        self.grid = [[self.make_cell(i, j) for i in range(self.width)] for j in range(self.height)]
        self.dictBackup = [[{} for _i in range(self.width)] for _j in range(self.height)]
        self.agents = []
        self.age = 0

    def make_cell(self, x, y):
        c = self.Cell()
        c.x = x
        c.y = y
        c.world = self
        c.agents = []
        return c

    def get_cell(self, x, y):
        return self.grid[y][x]

    def get_relative_cell(self, x, y):
        return self.grid[y % self.height][x % self.width]

    def load(self, f):
        if not hasattr(self.Cell, 'load'):
            return
        if isinstance(f, type('')):
            with open(f) as f:
                lines = f.readlines()
        else:
            lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        fh = len(lines)
        fw = max([len(x) for x in lines])

        if fh > self.height:
            fh = self.height
            start_y = 0
        else:
            start_y = (self.height - fh) // 2
        if fw > self.width:
            fw = self.width
            start_x = 0
        else:
            start_x = (self.width - fw) // 2

        self.reset()
        for j in range(fh):
            line = lines[j]
            for i in range(min(fw, len(line))):
                self.grid[start_y + j][start_x + i].load(line[i])

    def update(self, mouse_win=None, cat_win=None):
        if hasattr(self.Cell, 'update'):
            for a in self.agents:
                a.update()
            self.display.redraw()
        else:
            for a in self.agents:
                old_cell = a.cell
                a.update()
                if old_cell != a.cell:  # old cell won't disappear when new cell
                    self.display.redraw_cell(old_cell.x, old_cell.y)

                self.display.redraw_cell(a.cell.x, a.cell.y)

        if mouse_win:
            self.mouseWin = mouse_win
        if cat_win:
            self.catWin = cat_win
        self.display.update()
        self.age += 1

    def get_next_grid(self, x, y, dir):
        dx = 0
        dy = 0
        if self.directions == 8:
            dx, dy = [(0, -1), (1, -1), (
                1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)][dir]
        elif self.directions == 4:
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][dir]
        elif self.directions == 6:
            if y % 2 == 0:
                dx, dy = [(1, 0), (0, 1), (-1, 1), (-1, 0),
                          (-1, -1), (0, -1)][dir]
            else:
                dx, dy = [(1, 0), (1, 1), (0, 1), (-1, 0),
                          (0, -1), (1, -1)][dir]

        x2 = x + dx
        y2 = y + dy

        if x2 < 0:
            x2 += self.width
        if y2 < 0:
            y2 += self.height
        if x2 >= self.width:
            x2 -= self.width
        if y2 >= self.height:
            y2 -= self.height

        return x2, y2

    def add_agent(self, agent, x=None, y=None, cell=None, dir=None):
        self.agents.append(agent)
        if cell is not None:
            x = cell.x
            y = cell.y
        if x is None:
            x = random.randrange(self.width)
        if y is None:
            y = random.randrange(self.height)
        if dir is None:
            dir = random.randrange(self.directions)

        agent.cell = self.grid[y][x]
        agent.dir = dir
        agent.world = self


# GUI display
class tkinterDisplay_bis:
    def __init__(self, size=cfg.grid_width):
        self.activated = False
        self.paused = False
        self.title = ''
        self.updateEvery = 1
        self.root = None
        self.speed = cfg.speed
        self.bg = None
        self.size = size
        self.imageLabel = None
        self.frameWidth = 0
        self.frameHeight = 0
        self.world = None
        self.bg = None
        self.image = None

    def activate(self):
        if self.root is None:
            self.root = tkinter.Toplevel()
        for c in self.root.winfo_children():
            c.destroy()
        self.bg = None
        self.activated = True
        self.imageLabel = tkinter.Label(self.root)
        self.imageLabel.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)
        self.frameWidth, self.frameHeight = self.world.width * self.size, self.world.height * self.size
        self.root.geometry('%dx%d' % (self.world.width * self.size, self.world.height * self.size))
        self.root.update()
        self.redraw()
        self.root.bind('<Escape>', self.quit)

    def quit(self):
        self.root.destroy()

    def update(self):
        if not self.activated:
            return
        if self.world.age % self.updateEvery != 0 and not self.paused:
            return
        self.set_title(self.title)
        self.imageLabel.update()
        if self.speed > 0:
            time.sleep(float(1)/self.speed)

    def make_title(self, world):
        text = 'age: %d' % world.age
        extra = []
        if world.mouseWin:
            extra.append('mouseWin=%d' % world.mouseWin)
        if world.catWin:
            extra.append('catWin=%d' % world.catWin)
        if world.display.paused:
            extra.append('paused')
        if world.display.updateEvery != 1:
            extra.append('skip=%d' % world.display.updateEvery)
        if world.display.speed > 0:
            extra.append('speed=%dm/s' % world.display.speed)

        if len(extra) > 0:
            text += ' [%s]' % ', '.join(extra)
        return text

    def set_title(self, title):
        if not self.activated:
            return
        self.title = title
        title += ' %s' % self.make_title(self.world)
        if self.root.title() != title:
            self.root.title(title)

    def pause(self, event=None):
        self.paused = not self.paused
        while self.paused:
            self.update()

    def getBackground(self):
        if self.bg is None:
            r, g, b = self.imageLabel.winfo_rgb(self.root['background'])
            self.bg = b'%c%c%c' % (r >> 8, g >> 8, b >> 8)
        return self.bg

    def redraw(self):
        if not self.activated:
            return

        iw = self.world.width * self.size
        ih = self.world.height * self.size

        hexgrid = self.world.directions == 6
        if hexgrid:
            iw += self.size / 2

        with open('temp.ppm', 'wb') as f:
            f.write(b'P6\n%d %d\n255\n' % (iw, ih))

            odd = False
            for row in self.world.grid:
                line = io.StringIO()
                if hexgrid and odd:
                    line.write(self.getBackground() * (self.size / 2))
                for cell in row:
                    if len(cell.agents) > 0:
                        c = self.get_data_color(cell.agents[-1])
                    else:
                        c = self.get_data_color(cell)

                    line.write(c * self.size)
                if hexgrid and not odd:
                    line.write(self.getBackground() * (self.size / 2))
                odd = not odd

                f.write((line.getvalue() * self.size).encode('latin1'))
        # f.close()

        self.image = tkinter.PhotoImage(file='temp.ppm')
        self.imageLabel.config(image=self.image)

    imageCache = {}

    def redraw_cell(self, x, y):
        if not self.activated:
            return
        sx = x * self.size
        sy = y * self.size
        if y % 2 == 1 and self.world.directions == 6:
            sx += self.size / 2

        cell = self.world.grid[y][x]
        if len(cell.agents) > 0:
            c = self.get_text_color(cell.agents[-1])
        else:
            c = self.get_text_color(cell)

        sub = self.imageCache.get(c, None)
        if sub is None:
            sub = tkinter.PhotoImage(width=1, height=1)
            sub.put(c, to=(0, 0))
            sub = sub.zoom(self.size)
            self.imageCache[c] = sub
        self.image.tk.call(self.image, 'copy', sub, '-from', 0, 0, self.size, self.size, '-to', sx, sy)

    def get_text_color(self, obj):
        c = getattr(obj, 'color', None)
        if c is None:
            c = getattr(obj, 'color', 'white')
        if callable(c):
            c = c()
        if isinstance(c, type(())):
            if isinstance(c[0], type(0.0)):
                c = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            return '#%02x%02x%02x' % c
        return c

    dataCache = {}

    def get_data_color(self, obj):
        c = getattr(obj, 'color', None)
        if c is None:
            c = getattr(obj, 'color', 'white')
        if callable(c):
            c = c()
        if isinstance(c, type(())):
            if isinstance(c[0], type(0.0)):
                c = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            return '%c%c%c' % c
        else:
            val = self.dataCache.get(c, None)
            if val is None:
                r, g, b = self.imageLabel.winfo_rgb(c)
                val = '%c%c%c' % (r >> 8, g >> 8, b >> 8)
                self.dataCache[c] = val
            return val


def make_display_bis(world):
    d = tkinterDisplay_bis()
    d.world = world
    return d