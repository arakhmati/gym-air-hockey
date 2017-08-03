import numpy as np
import numbers

from abc import ABC, abstractmethod


class Vector(object):
    def __init__(self, v=np.zeros(2)):
        if isinstance(v, list):
            self.v = np.array(v, dtype=np.float)
        elif isinstance(v, np.array):
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
        self.v = np.zeros(2, dtype=np.float)
 
class BoundingShape(ABC):
    def __init__(self):
        pass

class BoundingCircle(BoundingShape):
    def __init__(self, **kwargs):
        if 'center' in kwargs and 'radius' in kwargs:
            self.center = kwargs['center']
            self.radius = kwargs['radius']
            
        elif 'circle_1' in kwargs and 'circle_2' in kwargs:
            circle_1 = kwargs['circle_1']
            circle_2 = kwargs['circle_2']
            center_offset = circle_2.center - circle_1.center
            distance = center_offset.square_magnitude()
            radius_difference = circle_2.radius - circle_1.radius
            
            # One encloses another one
            if radius_difference * radius_difference >= distance:
                if circle_1.radius > circle_2.radius:
                    self.center = circle_1.center
                    self.radius = circle_1.radius
                else:
                    self.center = circle_2.center
                    self.radius = circle_2.radius
            # Overlap
            else:
                distance = np.sqrt(distance)
                radius = (distance + circle_1.radius + circle_2.radius) * 0.5
                
                self.center = circle_1.center
                if distance > 0:
                    self.center += center_offset * ((radius - circle_1.radius) / distance)
                
    def overlaps(self, other):
        squared_distance = (self.center - other.center).squared_magnitude()
        return squared_distance < (self.radius + other.radius) * (self.radius + other.radius)
    
    def get_growth(self, other):
        new_circle = BoundingCircle(self, other)
        return new_circle.radius*new_circle.radius - self.radius*self.radius
    
    def get_size(self):
        return np.pi * self.radius * self.radius
        
class BoundingRectangle(BoundingShape):
    def __init__(self, left=0.0, top=0.0, width=0.0, height=0.0):
        
        self.left = left
        self.top = top
        self.width = width
        self.height = height
    
class RigidBody(ABC):
    def __init__(self, position):
        self.position = position
        self.velocity = Vector()
        self.acceleration = Vector()
        
        self._inverse_mass = 1.0
        self.friction = 0.995
        
        self.accumullated_forces = Vector()
        
    def set_position(self, position):
        self.position = position
        
    def set_velocity(self, velocity):
        self.velocity = velocity
        
    def set_acceleration(self, acceleration):
        self.acceleration = acceleration
    
    def set_mass(self, mass):
        if mass == 0.0:
            raise  ValueError('Mass cannot be zero')
        self.inverse_mass = 1 / mass
    
    def get_position(self):
        return self.position
    
    def get_velocity(self):
        return self.velocity
    
    def get_acceleration(self):
        return self.acceleration
    
    def get_inverse_mass(self):
        return self._inverse_mass
        
    def add_force(self, force):
        self.accumullated_forces += force
        
    # updates position and velocity
    def integrate(self, dt):
        
        last_frame_acceleration = self.acceleration + self.accumulated_forces * self._inverse_mass
        
        self.velocity += last_frame_acceleration * dt
        self.velocity *= np.pow(self.friction, dt)
        
        self.position += self.velocity * dt
        
    def clear_accumulators(self):
        self.accumullated_forces.clear()
        

    
        
class ForceGenerator(ABC):
    @abstractmethod
    def update_force(self, particle, dt):
        pass
    
#class Drag(ForceGenerator):
#    def __init__(self):
#        self.k1 = 0.995
#        self.k2 = 0.0
#        
#    def update_force(self, particle, dt):
#        force = particle.get_velocity()
#        drag_coefficient = force.magnitude()
#        drag_coefficient = self.k1*drag_coefficient + self.k2*drag_coefficient*drag_coefficient
#        
#        force.normalize()
#        force *= -drag_coefficient
#        particle.add_force(force)
    
class ForceRegistry(object):
    class Registry(object):
        def __init__(self, particle, force_generator):
            self.particle = particle
            self.force_generator = force_generator
    
    def __init__(self):
        self.registrations = set()
    
    def add(self, particle, force_generator):
        self.registrations.add(self.Registry(particle, force_generator))
        
    def remove(self, particle, force_generator):
        
        for r in self.registrations:
            if r.particle == particle and r.force_generator == force_generator:
                self.registrations.remove(r)
                break
                
    def clear(self):
        self.registrations = {}
        
    def update_forces(self):
        for r in self.registrations:
            r.force_generator.update_force(r.particle)
            
 
class PotentialContact(object):
    def __init__(self, bodies):
        self.bodies = bodies
           
class BoundingVolumeHierarchy(object):
    def __init__(self, parent, volume, body):
        self.parent = parent
        self.volume = volume
        self.body = body
        
        self.left_child  = None
        self.right_child = None
        
        if self.parent is None:
            if self.parent.left_child == self: 
                sibling = self.parent.right_child
            else:
                sibling = self.parent.left_child
                
        self.parent.volume = sibling.volume
        self.parent.body = sibling.body
        self.parent.left_child  = sibling.left_child
        self.parent.right_child = sibling.right_child
        
        self.parent.recalculate_bounding_volume()
        
        self.left_child  = None
        self.right_child = None
        
    def is_leaf(self):
        return self.body is not None
    
    def insert(self, new_volume, new_body):
        if self.is_leaf():
            self.left_child  = BoundingVolumeHierarchy(self, self.volume, self.body)
            self.right_child = BoundingVolumeHierarchy(self, new_volume, new_body)
            self.body = None
            self.recalculate_bounding_volume()
        else:
            if self.left_child.volume.get_growth() < self.right_child.volume.get_growth():
                self.left_child.insert(new_body, new_volume)
            else:
                self.right_child.insert(new_body, new_volume)
    
    def recalculate_bounding_volume(self, recurse=True):
        if self.is_leaf():
            return
        
        self.volume = BoundingCircle
    
                       
        
def Contact(object):
    def __init__(self, particles=(None, None), restitution=0.0, normal=Vector(), penetration=0.0):
        self.particles   = particles
        self.restitution = restitution
        self.normal      = Vector()
        self.penetration = penetration
        
    def compute_separating_velocity(self):
        relative_velocity = self.particles[0].get_velocity()
        if self.particles[1]: 
            relative_velocity -= self.particles[1].get_velocity()
        return relative_velocity * self.normal
    
    def _compute_total_inverse_mass(self):
        total_inverse_mass = self.particles[0].get_inverse_mass()
        if self.particles[1]: 
            total_inverse_mass += self. particles[1].get_inverse_mass()
            
        return total_inverse_mass
        
    def _resolve_velocity(self, dt):
        separating_velocity = self.compute_separating_velocity()
        if separating_velocity > 0.0:
            return
        
        new_separating_velocity = -separating_velocity * self.restiution
        
        velocity_due_to_acceleration = self.particles[0].get_acceleration()
        if self.particles[1]:
            velocity_due_to_acceleration -= self.particles[1].get_acceleration()
        separating_velocity_due_to_acceleration = velocity_due_to_acceleration * self.normal * dt
        
        if separating_velocity_due_to_acceleration < 0:
            new_separating_velocity += self.restitution * separating_velocity_due_to_acceleration
            
            new_separating_velocity = max(new_separating_velocity, 0.0)
        
        delta_velocity = new_separating_velocity - separating_velocity
        
        total_inverse_mass = self._compute_total_inverse_mass()
        if total_inverse_mass <= 0.0: 
            return
        
        impulse = delta_velocity / total_inverse_mass
        impulse_per_inverse_mass = self.normal * impulse
        
        self.particles[0].set_velocity(self.particle[0].get_velocity() + \
            impulse_per_inverse_mass * self.particles[0].get_inverse_mass())
        if self.particles[1]:
            self.particles[1].set_velocity(self.particles[1].get_velocity() + \
                impulse_per_inverse_mass * -self.particles[1].get_inverse_mass())
            
    def _resolve_interpenetration(self, dt):
        if self.penetration <= 0.0:
            return
        
        total_inverse_mass = self._compute_total_inverse_mass()
        if total_inverse_mass <= 0.0: 
            return
        
        disposition_per_inverse_mass = self.normal * (-self.penetration/total_inverse_mass)
        
        self.particles[0].set_velocity(self.particles[0].get_position() + \
            disposition_per_inverse_mass * self.particles[0].get_inverse_mass())
        if self.particles[1]:
            self.particles[1].set_velocity(self.particles[1].get_position() + \
                disposition_per_inverse_mass * self.particles[1].get_inverse_mass())
        
    def resolve(self, dt):
        self._resolve_velocity(dt)
        self._resolve_interpenetration(dt)
        
class ContactResolver(object):
    def __init__(self):
        self.iterations = 10
        self.iterations_used = 0
        
    def set_iterations(self, iterations):
        self.iterations = iterations
        
    def resolve_contacts(self, contacts, dt):
        self.iterations_used = 0
        while (self.iterations_used < self.iterations):
            max_index, max_value = max(contacts, key=lambda contact: contact.compute_separating_velocity())
            contacts[max_index].resolve(dt)
            self.iterations_used += 1
            
class World(object):
    def __init__(self):
        self.bodies = []
        
    def start_frame(self):
        for body in self.bodies:
            body.clear_accumulators()
            
    def run_physics(self, dt):
        for body in self.bodies:
            body.integrate(dt)
    
    
        
    