
class Device:
    def __init__(self, neur):
        self.neur = neur
        self.Gd = None
        self.Gndi = None

    def implant(self, Gnd):
        pass

    def record(self, Ar, nsteps):
        pass

    def stim(self, As, nsteps):
        pass

    def read(self, activity):
        pass

    def reset(self):
        pass

