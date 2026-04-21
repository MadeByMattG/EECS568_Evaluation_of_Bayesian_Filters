from rclpy.clock import Clock
from filter.PF import PF

class PF_ROS(PF):
    def __init__(self, system, init):
        super().__init__(system, init)

    def prediction(self, u , particles, step):
        particles = super().prediction(u, particles, step)
        self.state_.setTime(Clock().now().to_msg())
        return particles

    def correction(self, z, landmarks, particles, particle_weight, step):
        X, P, particles, particle_weight = super().correction(z, landmarks, particles, particle_weight, step)
        self.state_.setTime(Clock().now().to_msg())
        return X, P, particles, particle_weight

    def mean_variance(self):
        X, P  = super().mean_variance()
        self.state_.setTime(Clock().now().to_msg())
        return X, P

