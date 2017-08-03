import pygame
from pygame.constants import K_w, K_s, K_a, K_d

import math
import random
import numpy as np
import cv2

#import os
#os.environ['CUDA_VISABLE_DEVICE'] = ''
#print(os.environ['CUDA_VISABLE_DEVICE'])

colors = {
     'black': (   0,   0,   0),
     'white': ( 255, 255, 255),
     'green': (   0, 255,   0),
     'red':   ( 255,   0,   0),
     'blue':  (   0,   0, 255),
}

def distance(a, b):
    return np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2 )        

def angle(a, b):
    return norm_angle((math.atan2(b[1]-a[1],b[0]-a[0])*180)/math.pi)

def distance_from_O(a):
    return distance((0,0),a)        

def angle_from_O(a):
    return angle((0,0),a)    

def norm_angle(a):
    return (a+360) % 360

class Rink:
    
    def __init__(self, width, height, margin):
        self.__margin   = margin
        self.__left     = self.__margin
        self.__top      = self.__margin
        self.__right    = width - self.__margin
        self.__bottom   = height - self.__margin
        self.__width    = self.__right - self.__left
        self.__height   = self.__bottom - self.__top
        self.__center_x = width//2
        self.__center_y = height//2
        
    def get_margin(self):
        return self.__margin
    
    def get_left(self):
        return self.__left
    
    def get_top(self):
        return self.__top
    
    def get_right(self):
        return self.__right
    
    def get_bottom(self):
        return self.__bottom
    
    def get_width(self):
        return self.__width
    
    def get_height(self):
        return self.__height
    
    def get_center_x(self):
        return self.__center_x
    
    def get_center_y(self):
        return self.__center_y
    
    def draw(self, screen):
        pygame.draw.rect(screen, colors['white'], (self.__left, self.__top, self.__width, self.__height), 0)
        
        pygame.draw.line(screen,   colors['red'], [self.__left, self.__center_y], [self.__right, self.__center_y], 5)
        pygame.draw.circle(screen, colors['red'], [self.__center_x, self.__center_y], 50, 5)
        pygame.draw.circle(screen, colors['red'], [self.__center_x, self.__center_y], 10)
        
class GoalPosts(object):
    def __init__(self, center_x, bottom, margin, width=100, height=20):
        self.__center_x = center_x
        self.__bottom = bottom
        self.__width = width
        self.__height = height
        self.__top_y = (margin - self.__height)
        self.__bottom_y = self.__bottom - self.__height - self.__top_y
        
        self.__left = self.__center_x - self.__width/2        
        self.__right = self.__left + self.__width
        
        self.__score = {'top': 0, 'bottom': 0}
        self.__scored = False
        
    def get_left_post(self):
        return self.__left
    
    def get_right_post(self):
        return self.__right
    
    def update_score(self, post):
        self.__score[post] += 1
        self.__scored = True
        
    def scored(self):
        return self.__scored
    
    def get_score(self):
        return self.__score
    
    def reset(self):
        self.__scored = False
        
                
    def draw(self, screen):
        pygame.draw.rect(screen, colors['green'], (self.__left, self.__top_y,    self.__width, self.__height), 0)
        pygame.draw.rect(screen, colors['green'], (self.__left, self.__bottom_y, self.__width, self.__height), 0)

class Vector:

    def __init__( self, angle, magnitude):
        self.__angle = (angle+360)%360 
        self.__magnitude = magnitude
        
    def get_angle(self):
        return self.__angle

    def get_magnitude(self):
        return self.__magnitude
        
    def set_angle(self,angle):
        self.__angle = (angle+360)%360 
    def set_magnitude(self,magnitude):
        self.__magnitude = magnitude
        
    def get_xy(self):
        return math.cos(self.get_angle()*math.pi/180.0)*self.get_magnitude(),math.sin(self.get_angle()*math.pi/180.0)*self.get_magnitude()

    def set_xy(self,a):
        angle = angle_from_O(a)
        magnitude = distance_from_O(a)
        self.set_angle(angle)
        self.set_magnitude(magnitude)   

    def __add__(self,v):
        s1 = self.get_xy()
        s2 = v.get_xy()
        s3 = [s1[0]+s2[0],s1[1]+s2[1]]
        return Vector(angle( (0,0), s3),distance( (0,0), s3))

    def __sub__(self,v):
        s1 = self.get_xy()
        s2 = v.get_xy()
        s3 = [s1[0]-s2[0],s1[1]-s2[1]]
        return Vector(angle( (0,0), s3),distance( (0,0), s3))
        
    def __mul__( self, v):
        return self.get_magnitude()*v.get_magnitude()*math.cos(norm_angle(self.get_angle()-v.get_angle())*math.pi/180.0)
        
    def __rmul__( self, dt):
        return Vector(self.get_angle(),self.get_magnitude()*dt)
        
    def copy( self):
        return Vector(self.get_angle(),self.get_magnitude())
        
    def normalize( self):
        if self.get_magnitude()>0:
            self.set_magnitude(1)


class MovingCircle:

    def __init__(self, pos, rink, radius, mass, max_speed, friction):
        self.__pos = Vector(angle_from_O(pos), distance_from_O(pos))
        self.__start_pos_xy = pos
        self.__radius = radius
        self.__speed = Vector(0, 0)
        self.__mass = mass
        self.__max_speed = max_speed
        self.__friction = friction
        self.__rink = rink
        
    def get_pos(self):
        return self.__pos

    def get_pos_angle(self):
        return self.get_pos().get_angle()

    def get_pos_magnitude(self):
        return self.get_pos().get_magnitude()

    def get_pos_xy(self):
        return self.get_pos().get_xy()

    def get_radius(self):
        return self.__radius

    def get_speed(self):
        return self.__speed

    def get_speed_angle(self):
        return self.get_speed().get_angle()

    def get_speed_magnitude(self):
        return self.get_speed().get_magnitude()

    def get_speed_xy(self):
        return self.get_speed().get_xy()

    def get_mass(self):
        return self.__mass

    def get_max_speed(self):
        return self.__max_speed
    
    def get_friction(self):
        return self.__friction
    
    def get_rink(self):
        return self.__rink

    def set_pos(self,pos):
        self.__pos = pos

    def set_pos_angle(self,angle):
        self.get_pos().set_angle(angle)

    def set_pos_magnitude(self,magnitude):
        self.get_pos().set_magnitude(magnitude)

    def set_pos_xy(self,pos):
        self.get_pos().set_xy(pos)

    def set_radius(self,radius):
        self.__radius = radius

    def set_speed(self, speed):
        self.__speed = speed

    def set_speed_angle(self,angle):
        self.get_speed().set_angle(angle)

    def set_speed_magnitude(self,magnitude):
        self.get_speed().set_magnitude(magnitude)

    def set_speed_xy(self,speed):
        self.get_speed().set_xy(speed)

    def set_mass(self,m):
        self.__mass = m

    def set_max_speed(self,max_speed):
        self.__max_speed = max_speed 
        
    def get_start_pos_xy(self):
        return self.__start_pos_xy
    
    def set_friction(self,friction):
        self.__friction = friction
        
    def reset(self):
        pos = self.get_start_pos_xy()
        pos_angle = angle_from_O(pos)
        pos_magnitude = distance_from_O(pos)
        self.__pos = Vector(pos_angle,pos_magnitude)
        self.set_speed(Vector(0,0))        
                
    def friction( self, dt):
        if self.get_speed_magnitude()>0:
            self.set_speed_magnitude(self.get_speed_magnitude()-self.get_friction()*dt)
        if self.get_speed_magnitude()<0:
            self.set_speed_magnitude(0)
            self.set_speed_angle(0)

class Puck(MovingCircle):

    def __init__( self, pos, rink, goal, radius=15, mass=1, max_speed=0.6, friction=0.0001):
        super().__init__(pos, rink, radius, mass, max_speed, friction)
        self.__goal = goal
    
    def move(self, dt):
    
        self.friction(dt)
        
        new_pos = self.get_pos()+dt*self.get_speed()
        px, py = new_pos.get_xy()
        
        if px < self.get_rink().get_left()+self.get_radius():
            px = self.get_rink().get_left()+self.get_radius()
            self.set_speed_angle (180-self.get_speed_angle())
        elif px > self.get_rink().get_right()-self.get_radius():
            px = self.get_rink().get_right()-self.get_radius()
            self.set_speed_angle (180-self.get_speed_angle())
            
        if not (self.__goal.get_left_post()+self.get_radius()//2 < px < self.__goal.get_right_post()-self.get_radius()//2):
            if (py < self.get_rink().get_top()+self.get_radius()):
                py = self.get_rink().get_top()+self.get_radius()
                self.set_speed_angle(360-self.get_speed_angle())   
            elif (py > self.get_rink().get_bottom()-self.get_radius()):
                py = self.get_rink().get_bottom()-self.get_radius()
                self.set_speed_angle(360-self.get_speed_angle())
            
        self.set_pos_xy((px,py))
        
        if py < self.get_rink().get_top()-self.get_radius():
            self.__goal.update_score('bottom')
        elif py > self.get_rink().get_bottom()+self.get_radius():
            self.__goal.update_score('top')
            
         
    def collision(self, B, dt):
        A = self
        if A.get_speed_magnitude()==0:
            (A,B)=(B,A)

        S = A.get_speed()-B.get_speed()

        dist = distance( A.get_pos_xy(), B.get_pos_xy())
        sumRadii = A.get_radius() + B.get_radius()
        
        if dist > sumRadii:
            return False
        
        dist -= sumRadii

        if S.get_magnitude()*dt < dist:
            return False

        N = S.copy()
        N.normalize()
        C = B.get_pos()-A.get_pos()
        D = N*C

        if D <= 0:
            return False

        F = C.get_magnitude()**2-D**2

        sumRadiiSquared = sumRadii**2

        if F >= sumRadiiSquared :
            return False

        T = sumRadiiSquared - F

        if T<0:
            return False

        dist = D - math.sqrt(T)

        if S.get_magnitude()*dt < dist:
            return False

        # Collision happened
        N = C.copy()
        N.normalize()

        a1 = A.get_speed()*N
        a2 = B.get_speed()*N

        P = (2*(a1-a2))/(A.get_mass()+B.get_mass())
        newA = A.get_speed() - P*B.get_mass()*N
        newB = B.get_speed() + P*A.get_mass()*N
        
        A.set_speed(newA)
        B.set_speed(newB)
        
        if A.get_speed_magnitude()>A.get_max_speed():
            A.set_speed_magnitude(A.get_max_speed())
        if B.get_speed_magnitude()>B.get_max_speed():
            B.set_speed_magnitude(B.get_max_speed())
        
        return True
    
    def draw(self, screen):
        x, y = self.get_pos().get_xy()
        pygame.draw.circle(screen, colors['black'], [int(x), int(y)], self.get_radius(), 0)
        

class Mallet(MovingCircle):
    def __init__( self, player, rink, puck, mode='cpu', color=colors['red'], radius=20, mass=20, max_speed=0.6, friction=0.0001, acceleration=0.0025):
        
        self.__mode = mode
        self.__player = player
        if self.get_player() == 'top':
            pos = (rink.get_center_x(), rink.get_top()    + rink.get_margin())
        elif self.get_player() == 'bottom':
            pos = (rink.get_center_x(), rink.get_bottom() - rink.get_margin())
        else:
            raise ValueError("player can be either 'top' or 'bottom'")
            
        super().__init__(pos, rink, radius, mass, max_speed, friction)
        
        self.__acceleration = acceleration
        self.__puck = puck
        self.__color = color
        
        if self.get_player() == 'top':
            self.__top_bound    = self.get_rink().get_top() + self.get_radius()
            self.__bottom_bound = self.get_rink().get_top() + 0.5*(self.get_rink().get_height()) - self.get_radius()
        elif self.get_player() == 'bottom':
            self.__top_bound    = self.get_rink().get_top() + 0.5*(self.get_rink().get_height()) + self.get_radius()
            self.__bottom_bound = self.get_rink().get_top() + self.get_rink().get_height() - self.get_radius()
            
        self.__reachable_top_bound    = self.get_top_bound()    - self.get_radius()
        self.__reachable_bottom_bound = self.get_bottom_bound() + self.get_radius()
        self.__range = self.get_rink().get_width() // 4
        
        if mode == 'ai':
            from keras.models import load_model
            self.model = load_model('/home/ahmed/Documents/41X/model.h5')
            print(self.model.summary())
        
    def get_acceleration(self):
        return self.__acceleration

    def get_player(self):
        return self.__player
    
    def get_puck(self):
        return self.__puck
    
    def get_top_bound(self):
        return self.__top_bound
    
    def get_bottom_bound(self):
        return self.__bottom_bound
            
    def set_acceleration(self,acceleration):
        self.__acceleration = acceleration

    def set_player(self,player):
        self.__player = player
        
    def draw( self, screen):
        x, y = self.get_pos().get_xy()
        pygame.draw.circle(screen, self.__color,   [int(x), int(y)], self.get_radius(), 0)
        pygame.draw.circle(screen, colors['black'], [int(x), int(y)], self.get_radius(), 1)
        pygame.draw.circle(screen, colors['black'], [int(x), int(y)], 5,  0)
        
         
    def intersects(self, origin, direction, line):
        v1 = origin - line[0]
        v2 = line[1] - line[0]
        v3 = np.array([-direction[1], direction[0]])
        t1 = np.cross(v2, v1) / np.dot(v2, v3)
        t2 = np.dot(v1, v3) / np.dot(v2, v3)
        if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
            return [origin + t1 * direction]
        return None
    
    
    def overlaps(self):
        R0 = self.get_radius()
        R1 = self.get_puck().get_radius()
        x0, y0 = self.get_pos_xy()
        x1, y1 = self.get_puck().get_pos_xy()
        return (R0-R1)**2 <= (x0-x1)**2+(y0-y1)**2 <= (R0+R1)**2
    
    def move(self, dt, screen=None):
        
        
        touched = False
        self.friction(dt)
        old_px, old_py = self.get_pos_xy()
        puck = self.get_puck()
        puck_px, puck_py = puck.get_pos_xy()
        
        if self.__mode == 'cpu':
            px, py = old_px, old_py
            vx, vy = self.get_speed_xy()
            
            puck_vx, puck_vy = puck.get_speed_xy()
            
            if self.get_player() == 'top':
                goal_line = np.array([(200, 30), (300, 30)])
                
                
                
                intersects = self.intersects(np.array((puck_px, puck_py)), np.array((puck_vx, puck_vy)), goal_line)
                if intersects != None:
                    goal_px, goal_py = intersects[0]
                else:
                    goal_px, goal_py = (250, 30)
                
                x, y = 0, 0
                reachable = self.__reachable_top_bound <= puck_py <=  self.__reachable_bottom_bound
                if not reachable:
                    self.set_acceleration(0.0025)
                    x = random.randrange(-1, 2, 1)
                    y = random.randrange(-1, 2, 1)
                    
                    if x == -1 and px < self.get_rink().get_left()   + self.get_radius() + self.__range*2: x = 1
                    elif x == 1 and px > self.get_rink().get_right() - self.get_radius() - self.__range*2: x = -1
                    if y == 1 and py > self.get_top_bound() + self.__range: y = -1
                    
                else:
                    if puck_vy > 0:
                        self.set_acceleration(0.015)
                        if puck_px < px:
                            x = -1
                        if puck_px > px:
                            x = 1
                        if puck_py < py:
                            y = -1
                        if puck_py > py:
                            y = 1
                    else:
                        too_fast = puck.get_speed_magnitude() > 0.8 * puck.get_max_speed()
                        
                        if too_fast:
                            self.set_acceleration(0.008)
                            diff_px = goal_px - px
                            if abs(diff_px) < 5: x = 0
                            elif diff_px > 0:    x = 1
                            else:                x = -1
                            x *= min(abs(diff_px)/20, 1)
                        
                            diff_py = goal_py - py
                            if abs(diff_py) < 5: y = 0
                            elif diff_py > 0:    y = 1
                            else:                y = -1
                            y *= min(abs(diff_py)/20, 1)
                        else:
                            self.set_acceleration(0.015)
                            if puck_px < px:
                                x = -1
                            if puck_px > px:
                                x = 1
                            if puck_py < py:
                                y = -1
                            if puck_py > py:
                                y = 1
                
            else:
            
                goal_line = np.array([(200, 670), (300, 670)])
                
                intersects = self.intersects(np.array((puck_px, puck_py)), np.array((puck_vx, puck_vy)), goal_line)
                if intersects != None:
                    goal_px, goal_py = intersects[0]
                else:
                    goal_px, goal_py = (250, 670)
                
                x, y = 0, 0
                reachable = self.__reachable_top_bound <= puck_py <=  self.__reachable_bottom_bound
                if not reachable:
                    self.set_acceleration(0.0025)
                    x = random.randrange(-1, 2, 1)
                    y = random.randrange(-1, 2, 1)
                    
                    if x == -1 and px < self.get_rink().get_left()   + self.get_radius() + self.__range*2: x = 1
                    elif x == 1 and px > self.get_rink().get_right() - self.get_radius() - self.__range*2: x = -1
                    if y == -1 and py < self.get_bottom_bound() - self.__range: y = 1
                    
                else:
                    if puck_vy < 0:
                        self.set_acceleration(0.015)
                        if puck_px < px:
                            x = -1
                        if puck_px > px:
                            x = 1
                        if puck_py < py:
                            y = -1
                        if puck_py > py:
                            y = 1
                    else:
                        too_fast = puck.get_speed_magnitude() > 0.8 * puck.get_max_speed()
                        
                        if too_fast:
                            self.set_acceleration(0.008)
                            diff_px = goal_px - px
                            if abs(diff_px) < 5: x = 0
                            elif diff_px > 0:    x = 1
                            else:                x = -1
                            x *= min(abs(diff_px)/20, 1)
                        
                            diff_py = goal_py - py
                            if abs(diff_py) < 5: y = 0
                            elif diff_py > 0:    y = 1
                            else:                y = -1
                            y *= min(abs(diff_py)/20, 1)
                        else:
                            self.set_acceleration(0.015)
                            if puck_px < px:
                                x = -1
                            if puck_px > px:
                                x = 1
                            if puck_py < py:
                                y = -1
                            if puck_py > py:
                                y = 1
            
        elif self.__mode == 'keyboard':
            keys = pygame.key.get_pressed()  
            if   keys[pygame.K_a]: x = -1  
            elif keys[pygame.K_d]: x =  1
            else:                  x =  0                 
            if   keys[pygame.K_w]: y = -1       
            elif keys[pygame.K_s]: y =  1     
            else:                  y =  0
            
        elif self.__mode == 'ai':
            image = pygame.surfarray.array3d(screen)
            image = image[25:675, 25:475, :]
            image = cv2.resize(image, (128, 128))
            image = (image.astype(np.float)-128)/128
            image = image.reshape((1, 128, 128, 3))
            x, y = self.model.predict([image], batch_size=1)
            x = np.argmax(x) - 1
            y = np.argmax(y) - 1
            
            print(x, y)
            
        elif self.__mode == 'mouse':
            new_pos = pygame.mouse.get_pos()
            new_pos = Vector(angle_from_O(new_pos), distance_from_O(new_pos))
            new_px, new_py = new_pos.get_xy()
            px, py = self.get_pos_xy()
            vx = (new_px - px)/dt
            vy = (new_py - py)/dt
            
        else: # if AI just pass x, y from the arguments
            pass
        
        
        
        if self.__mode != 'mouse':
            vx,vy = self.get_speed_xy()
            vx += x*self.get_acceleration()*dt
            vy += y*self.get_acceleration()*dt
        
        self.set_speed_angle(angle_from_O((vx,vy)))
        new_speed = distance_from_O((vx,vy))
        if new_speed<=self.get_max_speed():
            self.set_speed_magnitude (new_speed)
 
        if self.__mode != 'mouse':
            new_pos = self.get_pos()+dt*self.get_speed()
        px,py = new_pos.get_xy()
        
        if (px < self.get_rink().get_left()+self.get_radius()):
            px = self.get_rink().get_left()+self.get_radius()
            touched = True
        elif (px > self.get_rink().get_right()-self.get_radius()):
            px = self.get_rink().get_right()-self.get_radius()
            touched = True
        
        if (py < self.get_top_bound()):
            py = self.get_top_bound()
            touched = True
        elif (py > self.get_bottom_bound()):
            py = self.get_bottom_bound()
            touched = True
            
        # Prevent from oscillating near the borders
        if touched:
            old_pos = self.get_pos_xy()
            self.set_speed_angle(angle(old_pos,(px,py)))
            self.set_speed_magnitude(distance(old_pos,(px,py))/dt)
            
        
        if self.overlaps():
            x = (old_px - puck_px) > 0
            px = old_px + x * 5
            y = (old_py - puck_py) > 0
            py = old_py + y * 5
        
        self.set_pos_xy((px, py))
        
        return x, y

class AirHockey():
    
    def __init__(self, width=500, height=700, margin=25, max_score=1):

        self.actions = {
            "left":  K_a,
            "right": K_d,
            "up":    K_w,
            "down":  K_s
        }
        
        self.__width  = width
        self.__height = height
        self.__margin = margin

        self.max_score = max_score
        
        self.score_sum = 0.0
        self.score_counts = {
            "top":    0.0,
            "bottom": 0.0
        }
        
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.get_screen_shape())
        
        self.__center_x = self.__width  // 2
        self.__center_y = self.__height // 2
        
        self.__rink = Rink(self.__width, self.__height, self.__margin)
        
        self.__goal_posts = GoalPosts(self.__center_x, self.__height, self.__margin)
        
        self.__puck = Puck((self.__center_x, self.__center_y), self.__rink, self.__goal_posts)
        
        self.__mallet_1  = Mallet('top',    self.__rink, self.__puck, mode='keyboard')
        self.__mallet_2  = Mallet('bottom', self.__rink, self.__puck, mode='ai')
        
        
        self._draw()
        
    
    
#    def _setAction(self, action, last_action):
#        """
#        Pushes the action to the pygame event queue.
#        """
#        if action is None:
#            action = self.NOOP
#
#        if last_action is None:
#            last_action = self.NOOP
#
#        kd = pygame.event.Event(KEYDOWN, {"key": action})
#        ku = pygame.event.Event(KEYUP, {"key": last_action})
#
#        pygame.event.post(kd)
#        pygame.event.post(ku)

    def get_screen_shape(self):
        return (self.__width, self.__height)

    def get_observations(self):
        return pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8)

    def get_actions(self):
        return self.actions.values()
        
    def get_state(self):
        return {}

    def get_score(self):
        return self.score_sum

    def episode_is_over(self):
        return (self.score_counts['top']    >= self.max_score) or \
               (self.score_counts['bottom'] >= self.max_score)
        
        
    def step(self, dt):

        self.__puck.collision(self.__mallet_1, dt)
        self.__puck.collision(self.__mallet_2, dt)
        
        self.__puck.move(dt)
        x, y = self.__mallet_1.move(dt)
        x, y = self.__mallet_2.move(dt, self.screen)
        
        self.screen.fill(colors['black'])
        if self.__goal_posts.scored():
            self._reset()
        self._draw()
        
##        # doesnt make sense to have this, but include if needed.
#        self.score_sum += self.rewards["tick"]
        
        if self.score_counts['top'] == self.max_score:
            self.score_sum += self.rewards["win"]

        if self.score_counts['bottom'] == self.max_score:
            self.score_sum += self.rewards["loss"]
            
    def _draw(self):
        self.screen.fill(colors['black'])
        for drawable in [self.__rink, self.__goal_posts, self.__puck, self.__mallet_1, self.__mallet_2]:
            drawable.draw(self.screen)

    def _reset(self):
        for obj in [self.__puck, self.__mallet_1, self.__mallet_2, self.__goal_posts]:
            obj.reset()

if __name__ == "__main__":

    pygame.init()
    game = AirHockey(width=500, height=700)

    done = False
    while not done:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        dt = game.clock.tick_busy_loop(50)
        game.step(dt)
        pygame.display.update()
    
    
    pygame.quit ()