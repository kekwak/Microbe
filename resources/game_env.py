import tkinter as tk
from random import randint, choice
import numpy as np
import os

class Settings:
    screen_width=500
    screen_height=500

    unit_size=30
    unit_lines=12
    unit_move_on_pixels=8
    unit_border_should_stop=True
    unit_default_name='Unit'

    coin_size=15
    coins_number=40
    coins_random_number=30
    coin_move_on_pixels=4
    coin_protection_radius=20

    line_size=90
    line_width=2
    line_default_name='Input'

    coin_types_colors={1: 'green', 2: 'yellow', 3: 'red'}
    coin_types_rewards={1: 25, 2: 10, 3: -100}

    allow_coin_types={1, 2, 3}

    border_padding=20
    use_distance_to_border=False
    print_intersection_info=False

class Unit:
    def __init__(self, canvas, name=Settings.unit_default_name):
        self.canvas = canvas
        self.screen_size = (Settings.screen_width, Settings.screen_height)
        self.size = Settings.unit_size
        self.coins_number = Settings.coins_number
        self.coin_random_number = Settings.coins_random_number
        self.unit_lines = Settings.unit_lines
        self.border_padding = Settings.border_padding
        line_default_name = Settings.line_default_name
        
        self.name = name
        self.score = 0

        self.position = self.screen_size[0] // 2 - self.size // 2, self.screen_size[1] // 2 - self.size // 2, \
            self.screen_size[0] // 2 + self.size // 2, self.screen_size[1] // 2 + self.size // 2
        
        self.lines = [
            Line(self.canvas, self.screen_size, angle, name=f'{line_default_name}_{number}')
            for number, angle in enumerate(np.arange(0, np.pi*2, np.pi/(self.unit_lines//2)))
        ]
        
        self.unit = self.canvas.create_oval(*self.position, fill='yellow')
        
        allow_coin_types = Settings.allow_coin_types & {1, 2, 3}
        coins_number = self.coins_number + randint(-self.coin_random_number, self.coin_random_number)
        self.coins = [
            Coin(self.canvas, self.screen_size, choice(tuple(allow_coin_types)), self.position)
            for _ in range(coins_number)
        ]

    def move(self, direction):
        if not self.canvas.winfo_exists():
            return
        
        unit_on_pixels = Settings.unit_move_on_pixels
        coin_on_pixels = Settings.coin_move_on_pixels
        print_intersection_info = Settings.print_intersection_info

        position = (direction[0]*unit_on_pixels, direction[1]*unit_on_pixels)
        self.canvas.move(self.unit, *position)
        for line in self.lines:
            self.canvas.move(line.line, *position)
        for coin in self.coins:
            coin_rand_dir = randint(0, 1)
            self.canvas.move(coin.coin, randint(-coin_on_pixels, coin_on_pixels)*(coin_rand_dir), randint(-coin_on_pixels, coin_on_pixels)*(1-coin_rand_dir))
        
        unit_coords = self.canvas.coords(self.unit)
        unit_center = ((unit_coords[0] + unit_coords[2]) / 2, (unit_coords[1] + unit_coords[3]) / 2)
        unit_radius = self.size / 2
        
        beg_score = self.score

        self.check_unit_position(unit_center, unit_radius)
        self.check_coins_collision(unit_center, unit_radius)
        self.check_line_coin_collision(unit_center, unit_radius)
        
        if print_intersection_info:
            lines_names = [line.name for line in self.lines]
            max_length = len(max(lines_names, key=len))
            formatted_lines = [f"{name:^{max_length+2}}" for name in lines_names]
            formatted_intersections = [f"{line.intersection:^{max_length+2}}" for line in self.lines]
            formatted_intersection_dists = [f"{line.intersection_dist:^{max_length+2}.2f}" for line in self.lines]

            print(f"|{'Name':^10}|{'Score':^10}|{'Step':^10}|" + '|'.join(formatted_lines) + '|')
            print(f"|{self.name:^10}|{self.score:^10}|{self.score-beg_score:^10}|" + '|'.join(formatted_intersections) + '|')
            print(f"|{'#'*32}|" + '|'.join(formatted_intersection_dists) + '|', end='\n'*2)
    
    def check_unit_position(self, unit_center, unit_radius):
        unit_x, unit_y = unit_center
        should_stop=Settings.unit_border_should_stop

        dir_x, dir_y = 0, 0
        if unit_x + unit_radius + self.border_padding < 0:
            dir_x = 1
        elif unit_x - unit_radius - self.border_padding > self.screen_size[0]:
            dir_x = -1
                
        if unit_y + unit_radius + self.border_padding < 0:
            dir_y = 1
        elif unit_y - unit_radius - self.border_padding > self.screen_size[1]:
            dir_y = -1
        
        if dir_x or dir_y:
            if should_stop:
                drection_x = (self.border_padding + unit_radius) * dir_x * (-1)
                drection_y = (self.border_padding + unit_radius) * dir_y * (-1)
            else:
                drection_x = (self.screen_size[0] + 2*self.border_padding + 2*unit_radius) * dir_x
                drection_y = (self.screen_size[1] + 2*self.border_padding + 2*unit_radius) * dir_y

            self.canvas.move(self.unit, drection_x, drection_y)
            for line in self.lines:
                self.canvas.move(line.line, drection_x, drection_y)

    def check_coins_collision(self, unit_center, unit_radius):
        for coin in self.coins:
            if not coin.is_alive:
                continue
            
            current_coin_coords = self.canvas.coords(coin.coin)
            current_coin_center = (current_coin_coords[0] + coin.size // 2, current_coin_coords[1] + coin.size // 2)
        
            dist = ((current_coin_center[0] - unit_center[0])**2 + \
                (current_coin_center[1] - unit_center[1])**2)**0.5
            
            if dist < (unit_radius + coin.size // 2) + 1:
                self.score += coin.kill()

    def check_line_coin_collision(self, unit_center, unit_radius):
        coin_types_colors=Settings.coin_types_colors
        use_distance_to_border = Settings.use_distance_to_border
        for line in self.lines:
            line_coords = self.canvas.coords(line.line)
            line_start = (line_coords[0], line_coords[1])
            line_end = (line_coords[2], line_coords[3])
            
            self.canvas.itemconfig(line.line, fill='black')
            line.intersection = 0
            line.intersection_dist = 0

            for coin in self.coins:
                if not coin.is_alive:
                    continue

                coin_coords = self.canvas.coords(coin.coin)
                coin_center = ((coin_coords[0] + coin_coords[2]) / 2, (coin_coords[1] + coin_coords[3]) / 2)
                coin_radius = coin.size / 2

                dist_to_line = self.distance_point_to_line(coin_center, line_start, line_end)
                dist_to_unit = self.distance_point_to_point(unit_center, coin_center)
                if dist_to_line - 2 <= coin_radius and (line.intersection_dist > dist_to_unit or line.intersection_dist == 0):
                    self.canvas.itemconfig(line.line, fill=coin_types_colors[coin.coin_type])
                    line.intersection = coin.coin_type
                    line.intersection_dist = dist_to_unit - unit_radius - coin_radius
                elif use_distance_to_border and self.is_out_of_border(line_start) and line.intersection == 0:
                    self.canvas.itemconfig(line.line, fill='blue')
                    line.intersection = 4
                    line.intersection_dist = self.distance_line_to_border(line_start, line_end) - unit_radius
                    

    def distance_point_to_line(self, point, line_start, line_end):
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        dx, dy = x2 - x1, y2 - y1
        det = dx * dx + dy * dy
        a = ((x - x1) * dx + (y - y1) * dy) / det

        a = max(0, min(1, a))
        nearest_x, nearest_y = x1 + a * dx, y1 + a * dy

        return ((x - nearest_x) ** 2 + (y - nearest_y) ** 2) ** 0.5

    def is_out_of_border(self, line_coords):
        return line_coords[0] < 0 or line_coords[1] < 0 or line_coords[0] > self.screen_size[0] or line_coords[1] > self.screen_size[1]

    def distance_point_to_point(self, point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def distance_line_to_border(self, point1, point2):
        if point2[0] - point1[0] == 0:
            point1 = point1[0] + 10**(-10), point1[1]

        a = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = (point2[1] - (a * point2[0]))

        x = min(max(point1[0], 0), self.screen_size[0])
        y = a * x + b if x != point1[0] else point1[1]
        
        y = min(max(y, 0), self.screen_size[1])
        x = (y - b) / a if y != point1[1] and a != 0 else x

        return self.distance_point_to_point((x, y), point2)
    
    def get_score(self):
        return self.score

class Line:
    def __init__(self, canvas, screen_size, angle, name='Input'):
        self.name = name
        
        self.intersection = 0
        self.intersection_dist = 0

        self.size = Settings.line_size
        self.width = Settings.line_width
            
        self.line = canvas.create_line(
            screen_size[0] // 2 + self.size * np.cos(angle),
            screen_size[1] // 2 - self.size * np.sin(angle),
            screen_size[0] // 2,
            screen_size[1] // 2,
            width=self.width,
        )

class Coin:
    def __init__(self, canvas, screen_size, coin_type, escape_square):
        self.canvas = canvas
        self.coin_type = coin_type
        self.screen_size = screen_size
        self.escape_square=escape_square

        self.size = Settings.coin_size
        self.protection_radius = Settings.coin_protection_radius
        
        self.is_alive = True
        
        self.coin_center = (randint(self.size, screen_size[0]-self.size), randint(self.size, screen_size[1]-self.size))
        while self.escape_square:
            if not (self.escape_square[0] - self.protection_radius <= self.coin_center[0] <= self.escape_square[2] + self.protection_radius and \
                self.escape_square[1] - self.protection_radius <= self.coin_center[1] <= self.escape_square[3] + self.protection_radius):
                break
            self.coin_center = (randint(self.size, screen_size[0]-self.size), randint(self.size, screen_size[1]-self.size))
        
        self.position = (self.coin_center[0] - self.size // 2, self.coin_center[1] - self.size // 2, \
            self.coin_center[0] + self.size // 2, self.coin_center[1] + self.size // 2)
        
        self.coin = self.canvas.create_oval(*self.position, fill=Settings.coin_types_colors[self.coin_type])

    def kill(self):
        self.canvas.delete(self.coin)
        self.is_alive = False
        return Settings.coin_types_rewards[self.coin_type]

class Game:
    def __init__(self, root=tk.Tk()):
        self.root = root
        self.screen_width = Settings.screen_width
        self.screen_height = Settings.screen_height
        self.canvas = tk.Canvas(root, width=Settings.screen_width, height=Settings.screen_height, bg='#DDD')
        self.canvas.pack()
        
        self.root.title('Бактерии')
        self.unit = Unit(self.canvas, Settings.unit_default_name)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_manually(self):
        self.root.bind('w', lambda _: self.unit.move((0, -1)))
        self.root.bind('a', lambda _: self.unit.move((-1, 0)))
        self.root.bind('s', lambda _: self.unit.move((0, 1)))
        self.root.bind('d', lambda _: self.unit.move((1, 0)))
    
        self.root.mainloop()

    def restart_game(self):        
        self.canvas.delete(self.unit.unit)
        for coin in self.unit.coins:
            self.canvas.delete(coin.coin)
        for line in self.unit.lines:
            self.canvas.delete(line.line)

        self.unit = Unit(self.canvas, (self.screen_width, self.screen_height))
    
    def on_close(self):
        os._exit(0)