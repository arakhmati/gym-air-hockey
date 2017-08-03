import numpy as np
import numbers

import pygame

from abc import ABC, abstractmethod


class Vector(object):
    def __init__(self, v=np.zeros(2)):
        self.n = len(v)
        if isinstance(v, list):
            self.v = np.array(v, dtype=np.float)
        elif isinstance(v, np.ndarray):
            self.v = v
        
    def magnitude(self):
        return np.linalg.norm(self.v)
    
    def squar_magnitude(self):
        return np.sum(np.transpose(self.v) * self.v)
    
    def normalize(self):
        mag = self.magnitude()
        
        if mag != 0.0:
            self.v /= mag
        
    def __add__(self, vector):
        return Vector(self.v + vector.v)
    
    def add(self, vector):
        self.v += vector.v
        
    def __sub__(self, vector):
        return Vector(self.v - vector.v)
    
    def sub(self, vector):
        self.v -= vector.v
        
    def __str__(self):
        return str(self.v[0]) + ' ' + str(self.v[1])
    
            
    def __mul__(self, b):
        if isinstance(b, numbers.Number): # scalar
            return Vector(self.v * b)
        elif isinstance(b, Vector):       # vector -> dot product
            return self.v.dot(b.v)
    
    def mul(self, scalar):
        self.v *= scalar
        
    def clear(self):
        self.v = np.zeros(self.n, dtype=np.float)
    
class Particle(ABC):
    def __init__(self, position, mass):
        self.position = Vector(position)
        self.velocity = Vector()
        self.acceleration = Vector()
        
        self.mass = mass
        self._inverse_mass = 1/float(mass)
        self.friction = 0.98
        self.wall_restitution = -1
        
        self.accumulated_forces = Vector()
        
    def set_position(self, position):
        self.position = position
        
    def set_velocity(self, velocity):
        self.velocity = velocity
        
    def set_acceleration(self, acceleration):
        self.acceleration = acceleration
    
    def set_mass(self, mass):
        self.mass = mass
        if mass == 0.0:
            raise  ValueError('Mass cannot be zero')
        self.inverse_mass = 1/mass
    
    def get_position(self):
        return self.position
    
    def get_velocity(self):
        return self.velocity
    
    def get_acceleration(self):
        return self.acceleration
    
    def get_inverse_mass(self):
        return self._inverse_mass
    
        
    def add_force(self, force):
        self.accumulated_forces += force
        
    def clear_accumulators(self):
        self.accumulated_forces.clear()        
        
    # updates position and velocity
    def integrate(self, dt):
        
        last_frame_acceleration = self.acceleration + self.accumulated_forces * self._inverse_mass
        
        self.velocity += last_frame_acceleration * dt
#        self.velocity *= np.power(self.friction, dt)
        self.velocity *= self.friction
        if self.velocity.magnitude() < 0.01:
            self.velocity = Vector([0, 0])
        
        self.position += self.velocity * dt
        
class Circle(Particle):
    def __init__(self, position, radius, mass, wall_restitution):
        super().__init__(position, mass)
        self.radius = radius
        self.wall_restitution = -wall_restitution
        
    def integrate(self, dt):
        
        super().integrate(dt)
        
        px, py = self.position.v
        if px < 25 + self.radius:
            px = 25 + self.radius
            self.velocity.v[0] *= self.wall_restitution
#            touched = True
        elif px > 475 - self.radius:
            px = 475 - self.radius
            self.velocity.v[0] *= self.wall_restitution
#            touched = True
        
        if py < 25 + self.radius:
            py = 25 + self.radius
            self.velocity.v[1] *= self.wall_restitution
#            touched = True
        elif py > 675 - self.radius:
            py = 675 -+ self.radius
            self.velocity.v[1] *= self.wall_restitution
            
        self.position = Vector([px, py])
        
class Disc(Circle):
    def __init__(self, position, radius, mass=1.0, wall_restitution=0.9):
        super().__init__(position, radius, mass, wall_restitution)
        
    def draw(self, screen):
        x, y = self.get_position().v
        pygame.draw.circle(screen, (0, 0, 0), [int(x), int(y)], self.radius, 0)
        
class Mallet(Circle):
    def __init__(self, position, radius, mass=15.0, wall_restitution=0.1, color=(255, 0, 0)):
        super().__init__(position, radius, mass, wall_restitution)
        import random
        self.color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        
    def draw(self, screen):
        x, y = self.get_position().v
        pygame.draw.circle(screen, self.color, [int(x), int(y)],  self.radius, 0)
        
class Plane():
    def __init__(self, normal, offset):
        self.normal = normal
        self.offset = offset
        
class ForceGenerator(ABC):
    @abstractmethod
    def update_force(self, rigid_body, dt):
        pass
    
class ConstantForce(ForceGenerator):
    def __init__(self, force):
        self.force = Vector(force)
        
    def update_force(self, rigid_body, dt):
        rigid_body.add_force(self.force)
        
class RandomForce(ForceGenerator):  
    def __init__(self, factor=1/15):
        self.factor = factor
        self.count = 0
        self.limit = 10
        self.force = Vector([0, 0])
        
    def update_force(self, rigid_body, dt):
        import random
        if self.count == self.limit:
           self.force =  Vector([random.randrange(-1, 2, 1)*self.factor, random.randrange(-1, 2, 1)*self.factor])
           self.count = 0
        self.count += 1
        rigid_body.add_force(self.force)
        
class KeyboardForce(ForceGenerator):  
    def __init__(self, factor=1/20, player=0):
        self.factor = factor
        self.player = player
        
        
    def update_force(self, rigid_body, dt):
        if self.player == 0:
            keys = pygame.key.get_pressed()  
            if   keys[pygame.K_a]: x = -1  
            elif keys[pygame.K_d]: x =  1
            else:                  x =  0                 
            if   keys[pygame.K_w]: y = -1       
            elif keys[pygame.K_s]: y =  1     
            else:                  y =  0
        else:
            keys = pygame.key.get_pressed()  
            if   keys[pygame.K_LEFT]:  x = -1  
            elif keys[pygame.K_RIGHT]: x =  1
            else:                      x =  0                 
            if   keys[pygame.K_UP]:    y = -1       
            elif keys[pygame.K_DOWN]:  y =  1     
            else:                      y =  0
            
        self.force =  Vector([x*self.factor, y*self.factor])
        rigid_body.add_force(self.force)
    
class ForceRegistry(object):
    class Registry(object):
        def __init__(self, rigid_body, force_generator):
            self.rigid_body = rigid_body
            self.force_generator = force_generator
    
    def __init__(self):
        self.registrations = set()
    
    def add(self, rigid_body, force_generator):
        self.registrations.add(self.Registry(rigid_body, force_generator))
        
    def remove(self, rigid_body, force_generator):
        for registration in self.registrations:
            if registration.particle == rigid_body and registration.force_generator == force_generator:
                self.registrations.remove(registration)
                break
        
    def update_forces(self, dt):
        for registration in self.registrations:
            registration.force_generator.update_force(registration.rigid_body, dt)
                
    def clear(self):
        self.registrations = {}
 
class CollisionDetector(object):

          
        
class Contact(object):
    def __init__(self, bodies, normal, point, penetration, restitution):
        self.particles   = bodies
        self.normal      = normal
        self.point       = point
        self.penetration = penetration
        self.restitution = restitution
    
        
    def compute_separating_velocity(self):
        relative_velocity = self.particles[0].get_velocity()
        if self.particles[1]: 
            relative_velocity -= self.particles[1].get_velocity()
        return relative_velocity * self.normal
    
    def _compute_total_inverse_mass(self):
        total_inverse_mass = self.particles[0].get_inverse_mass()
        if self.particles[1]: 
            total_inverse_mass += self.particles[1].get_inverse_mass()
        return total_inverse_mass
        
    def _resolve_velocity(self, dt):
#        separating_velocity = self.compute_separating_velocity()
#        print('%30s %20s %10f' % ('_resolve_velocity', 'separating_velocity', separating_velocity))
#        
#        if separating_velocity >= 0.0:
#            return
#
#        if self.particles[0].velocity.magnitude() == 0.0 or self.particles[1].velocity.magnitude() == 0.0:
#            self.normal *= -1
#        
#        print('%30s %20s %10f' % ('_resolve_velocity', 'self.restitution', self.restitution))
#        new_separating_velocity = -separating_velocity * self.restitution
#        
#        velocity_due_to_acceleration = self.particles[0].get_acceleration()
#        if self.particles[1]:
#            velocity_due_to_acceleration -= self.particles[1].get_acceleration()
#        separating_velocity_due_to_acceleration = velocity_due_to_acceleration * self.normal * dt
#        
#        if separating_velocity_due_to_acceleration < 0:
#            new_separating_velocity += self.restitution * separating_velocity_due_to_acceleration
#            new_separating_velocity = max(new_separating_velocity, 0.0)
#        
#        delta_velocity = new_separating_velocity - separating_velocity
#        
#        total_inverse_mass = self._compute_total_inverse_mass()
#        print('%30s %20s %10f' % ('_resolve_velocity', 'total_inverse_mass', total_inverse_mass))
#        if total_inverse_mass <= 0.0: 
#            return
#        
#        impulse = delta_velocity / total_inverse_mass
#        impulse_per_inverse_mass = self.normal * impulse
#        
#        self.particles[0].set_velocity(self.particles[0].get_velocity() + impulse_per_inverse_mass * self.particles[0].get_inverse_mass())
#        if self.particles[1]:
#            self.particles[1].set_velocity(self.particles[1].get_velocity() + impulse_per_inverse_mass * -self.particles[1].get_inverse_mass())

        circles = self.particles
        if (circles[0].velocity - circles[1].velocity) * (circles[0].position - circles[1].position) > 0:
            return
#            
        newVelX1 = (circles[0].velocity.v[0] * (circles[0].mass - circles[1].mass) + (2 * circles[1].mass * circles[1].velocity.v[0])) / (circles[0].mass + circles[1].mass)
        newVelY1 = (circles[0].velocity.v[1] * (circles[0].mass - circles[1].mass) + (2 * circles[1].mass * circles[1].velocity.v[1])) / (circles[0].mass + circles[1].mass)
        newVelX2 = (circles[1].velocity.v[0] * (circles[1].mass - circles[0].mass) + (2 * circles[0].mass * circles[0].velocity.v[0])) / (circles[0].mass + circles[1].mass)
        newVelY2 = (circles[1].velocity.v[1] * (circles[1].mass - circles[0].mass) + (2 * circles[0].mass * circles[0].velocity.v[1])) / (circles[0].mass + circles[1].mass)
        
        circles[0].position.v[0] = circles[0].position.v[0] + newVelX1 * dt;
        circles[0].position.v[1] = circles[0].position.v[1] + newVelY1 * dt;
        circles[1].position.v[0] = circles[1].position.v[0] + newVelX2 * dt;
        circles[1].position.v[1] = circles[1].position.v[1] + newVelY2 * dt;
        
        circles[0].velocity = Vector([newVelX1, newVelY1])
        circles[1].velocity = Vector([newVelX2, newVelY2])
    
            
    def _resolve_interpenetration(self, dt):
        print('%30s %20s %10f' % ('_resolve_interpenetration', 'self.penetration', self.penetration))
        if self.penetration <= 0.0:
            return
        
        total_inverse_mass = self._compute_total_inverse_mass()
        if total_inverse_mass <= 0.0: 
            return
        print('%30s %20s %10f' % ('_resolve_interpenetration', 'total_inverse_mass', total_inverse_mass))
        
        disposition_per_inverse_mass = self.normal * (self.penetration / total_inverse_mass)
        
        self.particles[0].set_position(self.particles[0].get_position() + \
            disposition_per_inverse_mass * self.particles[0].get_inverse_mass())
        if self.particles[1]:
            self.particles[1].set_position(self.particles[1].get_position() + \
                disposition_per_inverse_mass * -self.particles[1].get_inverse_mass())
        
    def resolve(self, dt):
        self._resolve_velocity(dt)
        self._resolve_interpenetration(dt)
        
class World(object):
    def __init__(self):
        self.objects = [Disc([250, 350], 25), Disc([375, 375], 25), Disc([375, 125], 25), Mallet([250, 125], 25, color=(0, 255, 0)), Mallet([250, 575], 25), Mallet([125, 575], 25), Mallet([125, 125], 25)]
        
        self.forces = ForceRegistry()
        self.forces.add(self.objects[3], KeyboardForce(1/50))
        for obj in self.objects[4:]:
            self.forces.add(obj, RandomForce(1/50))

    def detect_collision(self, circle_1, circle_2, restitution=0):
        position_1 = circle_1.get_position()
        position_2 = circle_2.get_position()
        
        middle = position_1 - position_2
        distance = middle.magnitude()
        
        if distance <= 0.0 or distance >= (circle_1.radius + circle_2.radius):
            return (False, None)
        
#        print('position_1 %f %f' % (position_1.v[0], position_1.v[1]))
#        print('position_2 %f %f' % (position_2.v[0], position_2.v[1]))
#        print('middle %f %f' % (middle.v[0], middle.v[1]))
#        print('distance %f\n' % distance)
        
        normal = middle * (1.0/distance)
        point = position_1 + middle*0.5
        penetration = circle_1.radius + circle_2.radius - distance
        
        contact = Contact((circle_1, circle_2), normal, point, penetration, restitution)
        return (True, contact) 
            
    def update(self, dt, screen):
        
        for obj in self.objects:
            obj.clear_accumulators()
            
        self.forces.update_forces(dt)
        
        for obj in self.objects:
            obj.draw(screen)
            obj.integrate(dt)
            
        contacts = []
        for i in range(len(self.objects)):
            for j in range(i, len(self.objects)):
                if i == j: continue
                collision, contact = self.collision_detector.circle_and_circle(self.objects[i], self.objects[j])
                if collision:  contacts.append(contact)
                
        for i in range(len(contacts)):
            contacts[i].resolve(dt)
                    
            
    
    
if __name__ == "__main__":

    pygame.init()
    
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((500, 700))
    
    env = World()

    done = False
    while not done:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        dt = clock.tick_busy_loop(60)
        
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (25, 25, 450, 650), 0)
        env.update(dt, screen)
        
        pygame.display.update()
    
    
    pygame.quit ()       
    