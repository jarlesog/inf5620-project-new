# Begin demo

from dolfin import *
import time
time0 = time.time()

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
mesh = UnitSquare(4,32)

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Set parameter values
dt = 0.0001
T = 0.1

nu = 1.0
A = 1; 
a = 1;

u_right = Expression(("sin(pi*x[1])*A*exp(a*t)", "0"), a = a, A=A, t = 0.0)
u_left =  Expression(("sin(pi*x[1])*A*exp(a*t)", "0"), a = a, A=A, t = 0.0)

# Define boundary conditions
rightline = DirichletBC(V, u_right, "on_boundary && x[0] > 1 - DOLFIN_EPS")
leftline = DirichletBC(V, u_left, "on_boundary && x[0] < DOLFIN_EPS")
topline = DirichletBC(V, (0,0), "on_boundary && x[1] > 1 - DOLFIN_EPS")
bottomline = DirichletBC(V, (0,0), "on_boundary && x[1] < DOLFIN_EPS")

ptop   = DirichletBC(Q, 0, "x[1] > 1.0 - DOLFIN_EPS")
pright = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")
pbottom= DirichletBC(Q, 0, "x[1] < DOLFIN_EPS")
pleft  = DirichletBC(Q, 0, "x[0] < DOLFIN_EPS")
bcu = [bottomline, rightline, topline, leftline]
bcp = [ptop, pright, pbottom, pleft]

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)
u_e = Expression(("A*exp(a*t)*sin(pi*x[1])","0"),  a = a, A=A, t = 0.0)
u0.assign(u_e)

# Define coefficients
k = Constant(dt)
class MyExpression(Expression):    
    
    def __init__(self, t = 0.0, A = A, a = a):
        self.A = A
        self.a = a
        self.t = t
        Expression.__init__(self)
    
    def eval(self, values, x):
        tmp_test = pi*self.A*self.A*exp(2*self.a*self.t)*sin(pi*x[1])*cos(pi*x[1])
        values[0] = (pi*pi + self.a)*self.A*exp(self.a*self.t)*sin(pi*x[1])
        values[1] = 0.0;
    
    def value_shape(self):
        return (2,)
f = MyExpression();#

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u)*u0, v)*dx + \
     nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx #Semi implicit
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
#a2 = inner(grad(p), grad(q))*dx
#L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
#A2 = assemble(a2)
A3 = assemble(a3)


# Create files for storing solution
#ufile = File("results/velocity.pvd")
#pfile = File("results/pressure.pvd")

# Time-stepping
t = dt    
error_tot = 0
n_step = 0
while t < T + DOLFIN_EPS:

    # Update pressure boundary condition
    f.t = t
    u_e.t = t
    u_left.t = t
    u_right.t = t
    # Compute tentative velocity step
    begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "gmres", "default")
    end()

    # Pressure correction
    #begin("Computing pressure correction")
    #b2 = assemble(L2)
    #[bc.apply(A2, b2) for bc in bcp]
    #solve(A2, p1.vector(), b2, "gmres", "amg")
    #end()

    # Velocity correction
    begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "gmres", "default")
    end()

    print "The Error Norm is:"
    E = errornorm(u_e, u1, degree=3)
    error_tot += E
    n_step += 1
    print E
    
    # Plot solution
    #plot(p1, title="Pressure", rescale=True)
    plot(u1, title="Velocity", rescale=True)
    #plot(u_e - u1,  title="Error", rescale=True)
    # Save to file
    #ufile << u1
    #pfile << p1
    

    # Move to next time step
    u0.assign(u1)
    t += dt
    print "t =", t

print 'The avrage error: ',error_tot/n_step
print 'Program used %g sec.' % (- time0 + time.time())

# Hold plot
interactive()

#y-axis: 4#The avrage error:  0.048635
#y-axis: 8#The avrage error:  0.0123844486516
#y-axis:16#The avrage error:  0.00310446157346
#y-axis:32#The avrage error:  0.000775229894969
