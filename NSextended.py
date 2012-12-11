# Begin demo

from dolfin import *
import time
time0 = time.time()

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
mesh = Mesh("lshape.xml.gz")

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
phi=TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)
psi=TestFunction(Q)

# Set parameter values
dt = 0.01
T = 3

nu = 0.01
C = 10. #Diffution constant


# Define time-dependent pressure boundary condition
#p_in = Expression("2000*sin(3.0*t)", t=0.0)
class MyExpression(Expression):
    #This is a sinus wave that is zero if negative.
    def __init__(self, t = 0.0):
        self.t = t
        Expression.__init__(self)

    def eval(self, values, x):
        tmp_value = 2000*sin(3.0*self.t)
        if tmp_value < 0.0:
            values[0] = 0.0
        else: values[0] = tmp_value
p_in = MyExpression();


# Define the rate function
#R = Expression("(3*(1-x[0])*x[0]*(1-x[1])*x[1])*(t + 1)", t = 0.0)
R = Constant(0)



# Define boundary conditions
noslip  = DirichletBC(V, (0, 0),
                      "on_boundary && \
                       (x[0] < DOLFIN_EPS | x[1] < DOLFIN_EPS | \
                       (x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS))")
inflow  = DirichletBC(Q, p_in, "x[1] > 1.0 - DOLFIN_EPS")
outflow = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")
bcu = [noslip]
bcp = [inflow, outflow]

# Define boundray conditions on phi
noslipphi = DirichletBC(Q, 0, "on_boundary && \
                            (x[0] < DOLFIN_EPS | x[1] < DOLFIN_EPS | \
                                 (x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS))")
inflowphi = DirichletBC(Q, 1, "x[1] > 1.0 - DOLFIN_EPS")
outflowphi = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")
bcphi = [noslipphi, inflowphi, outflowphi]


# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)
phi0=Function(Q)
#phi1 = Expression("1.0")#Constant(1)
#phi0.assign(phi1)


# Define coefficients
k = Constant(dt)
f = Constant((0, 0))

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u)*u0, v)*dx + \
     nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx #Semi implicit
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)


# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")
phifile=File("results/phi.pvd")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:

    # Update pressure boundary condition
    p_in.t = t
    R.t = t
    
    # Compute tentative velocity step
    begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "gmres", "default")
    end()
    
    # Pressure correction
    begin("Computing pressure correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2, "gmres", "amg")
    end()
    
    # Velocity correction
    begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "gmres", "default")
    end()
    
    #Phi solveing
    ##Assembleing
    F4 = (1/k)*inner(phi-phi0, psi)*dx + inner(dot(u1,grad(phi)),psi)*dx \
    + C*inner(grad(phi),grad(psi))*dx + inner(R*phi, psi)*dx#Is this implicit?
    a4 = lhs(F4)
    L4 = rhs(F4)
    A4 = assemble(a4)
    b4 = assemble(L4)
    ##
    
    begin("Computeing phi")
    [bc.apply(A4, b4) for bc in bcphi]
    solve(A4, phi0.vector(), b4, "gmres", "amg")
    end()
    
    # Plot solution
    #plot(p1, title="Pressure", rescale=True)
    #plot(u1, title="Velocity", rescale=True)
    #plot(phi0,title="phi    ",rescale=True)
    
    # Save to file
    ufile << u1
    pfile << p1
    phifile<<phi0

    # Move to next time step
    u0.assign(u1)
    t += dt
    print "t =", t


print 'Program used %g sec.' % (- time0 + time.time())

# Hold plot
interactive()

