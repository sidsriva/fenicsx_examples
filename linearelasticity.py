# Scaled variable
#import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
L = 1
W = 0.2
mu = 1
rho = 1
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma
#Mesh
#domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],
#                         [20, 6, 6], cell_type=mesh.CellType.hexahedron)
#domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],
                        #  [20, 6, 6], cell_type=mesh.CellType.tetrahedron)

# domain=mesh.Mesh()
# with io.XDMFFile('./brick_w_holes/mesh.xdmf') as infile:
# 	infile.read(domain)
with io.XDMFFile(MPI.COMM_WORLD, './brick_w_holes/mesh.xdmf', "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")

V = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
#Dirichlet boundary conditions
def clamped_boundary_1(x):
    return np.isclose(x[2], -12.5)
def clamped_boundary_2(x):
    return np.isclose(x[2], 12.5)


fdim = domain.topology.dim - 1
clamped_boundary1_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary_1)
clamped_boundary2_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary_2)


u1_D = np.array([0, 0, 0], dtype=default_scalar_type)
u2_D = np.array([5, 0, 0], dtype=default_scalar_type)

bc1 = fem.dirichletbc(u1_D, fem.locate_dofs_topological(V, fdim, clamped_boundary1_facets), V)
bc2 = fem.dirichletbc(u2_D, fem.locate_dofs_topological(V, fdim, clamped_boundary2_facets), V)

#Neumann boundary conditions
T = fem.Constant(domain, default_scalar_type((0, 0, 0))) #Homogeneous Neumann condition


#Boundary integral
ds = ufl.Measure("ds", domain=domain)

#Weak form
def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, 0)))        #no body force
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds      #Homogeneous Neumann Boundary condition
#Solver
problem = LinearProblem(a, L, bcs=[bc1, bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()


#Paraview plotting
with io.XDMFFile(domain.comm, "result/deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"
    xdmf.write_function(uh)
