frames = 250
res = 200
gs = vec3(res,res,res)
s = Solver(name='main', gridSize = gs)
s.timestep = 1.0

flags = s.create(FlagGrid)
vel = s.create(MACGrid)
density = s.create(RealGrid)
pressure = s.create(RealGrid)

# phiObs   = s.create(LevelsetGrid)

obstacle = s.create(Sphere, center=gs*vec3(0.5,0.5,0.5), radius=res*0.1, z=gs*vec3(0, 0.5, 0))
phiObs = obstacle.computeLevelset()


flags.initDomain()
setObstacleFlags(flags=flags, phiObs=phiObs) #, fractions=fractions)
flags.fillGrid()

source = s.create(Cylinder, center=gs*vec3(0.5,0.1,0.5), radius=res*0.14, z=gs*vec3(0, 0.02, 0))

if (GUI):
    gui = Gui()
    gui.show()
    gui.pause()

for t in range(frames):
    mantaMsg('Frame {}'.format(s.frame))
    source.applyToGrid(grid=density, value=1)

    advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2)
    advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=2)

    setWallBcs(flags=flags, vel=vel)
    addBuoyancy(density=density, vel=vel, gravity=vec3(0,-6e-4,0), flags=flags)
    solvePressure(flags=flags, vel=vel, pressure=pressure)
    s.step()
