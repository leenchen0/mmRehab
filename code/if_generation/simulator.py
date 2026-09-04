import time
import numpy as np

class Simulator:
    """The main simulator class"""
    def __init__(self, radars, scene, period: float, fps: int):
        """
        Parameters:
            radars: a list of radar objects.
            scene: a list of scene objects.
            period: duty time in a frame.   (n_tx * n_loop)(idle time + ramp end time)
            fps: n_loop/period  Note that the assumption that time invariant within a loop is necessary holds true
        """
        self.T      = period
        self.fps    = fps
        self.radars = radars
        self.scene  = scene
        for obj in self.scene:
            obj.register(self.T, self.fps)      # register the frame information for each object
        self.name = f'{self.__class__.__name__}'
        self.max_v = 1e-3*fps                   # maximum allowed velocity to avoid ambigious phase
        # print(f'[{self.name}] Allowed maximum velocity is {self.max_v} m/s')

    def run(self):
        """Run the simulation and return the simulated data

        Return:
            simulated radar siganl matrix of shape (n_rx, n_chirp, n_sample).
        """
        signals = []
        for radar in self.radars:   
            # simulate for each rx
            signal, __ = radar.my_reflect_motion_multi(self.scene)
            signals.append(signal)
        signals = np.array(signals)

        return signals

    def get_paths(self):
        """Return the path of all points in the scene"""
        res = []
        for obj in self.scene:
            res.append(obj.get_path())
        return res