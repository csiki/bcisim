
class Device:
    def __init__(self, neur):
        self.neur = neur
        self.Gd = None
        self.Gndi = None
        self.Gdni = None

    def implant(self, Gnd):
        # Gnd is neur->dev connection graph or dictionary:
        self.Gndi = Gnd

        # build a reverse Gndi = Gdni
        # self.Gdni = {for n, d in self.Gndi.items() } TODO

    def record(self, Ar, nsteps):
        # Ar is a binary vector index by device recording sites
        # self.neur.record(self.Gndi, sites, nsteps)
        pass  # TODO

    def stim(self, As, nsteps):
        pass

    def read(self, activity):
        pass

    def reset(self):
        pass


if __name__ == '__main__':
    pass
