# --------------------------------
# Thermoelasticity with coupled heat transfer
# Author: Siddhartha Srivastava (sidsriva@umich.edu)
# Date: April 01, 2024
# Uses following docker image: quay.io/fenicsproject/stable:current
# Fields of interest: 
#   (u, T) in \Omega \subset \mathbb{R}^3
# Governing equation:
#    \nabla\cdot\sigma = 0
#    \frac{\partial T}{\partial t} = \nabla\cdot \kappa \nabla T + g
# Kinematic relationship
#   \epsilon = 1/2 (\nabla u + \nabla u^T)
# Constitutive relationship
#    \epsilon_e = \epsilon - \alpha T \mathbb{I}
#    \sigma = \lambda trace(\epsilon_e) \mathbb{I} + 2 \mu \epsilon_e
# Domain: 
#    A box (10 cm x 10 cm x 2 cm) with a cylindrical hole (along z axis) of radius 1cm at the center
# Boundary conditions:
#    Noting the dimensions of the box are [-5, 5] x [-5, 5] x [-1, 1]
#    u(x =-5) = 0
#    v(y =-5) = 0
#    u(y = 5) = top_disp(t)
#    T(x =-5) = 0
#    T(x = 5) = 100
#    Homogenous Neumann BC on the rest of the boundaries
#-------------------------------------
from ufl import *
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
set_log_active(True)
# import principal_components as pc

#------------------------------------------------
#          Define Parameters
#------------------------------------------------

## Loading conditions
Max_y_disp = 0.001 # Net displ (m) 
t_max = 1 # Total time for displacing by Max_y_disp (s)
numTSteps = 11 # Time steps
tau = Constant((t_max-0)/(numTSteps-1)) #Timestep size (gets updated in the inner loop )


## Parameters for aluminium
rho = 2.6 #density(Mg/m^3) - Not used
E = 70 #GPa
nu = 0.35 #unitless
alpha = 22e-6 #Coeff of thermal expansion (per K)
kappa = 237.  #Thermal conductivity (W/m K)


## Mesh refinement
mesh_elem_num = 20

## Integration
q_degree = 3


#------------------------------------------------
#          Define Geometry
#------------------------------------------------

# Define mesh
from dolfin import *
from mshr import *
#Length unit = m
domain = Box(Point(-0.05,-0.05,-0.01),Point(0.05,0.05,0.01)) - Cylinder(Point(0, 0, -0.01), Point(0, 0, 0.01), 0.01, 0.01)
mesh=generate_mesh(domain,mesh_elem_num)

# mesh = Mesh()
# with XDMFFile("./mesh_files/trimesh.xdmf") as file:
#     file.read(mesh)

dx_ = dx(metadata={'quadrature_degree': q_degree})
x = SpatialCoordinate(mesh)
ndim = mesh.topology().dim()


#------------------------------------------------
#          Define Space
#------------------------------------------------

Pu = VectorElement('Lagrange', mesh.ufl_cell(), 1) #For displacements
PT = FiniteElement('Lagrange', mesh.ufl_cell(), 1) #For relative temperature change
element = MixedElement([Pu,PT]) 
V = FunctionSpace(mesh, element)

V_u = VectorFunctionSpace(mesh, 'Lagrange', 1)
V_T = FunctionSpace(mesh, 'Lagrange', 1)
V_stress = TensorFunctionSpace(mesh, 'DG', 0)
#For projecting solution
fa_lift = FunctionAssigner(V, [V_u,V_T] )


# Define test and trial functions
w = TestFunction(V)
w_u, w_T = split(w)

# Define functions for variables
state = Function(V)
state_n = Function(V)
stress_print = Function(V_stress, name='Stress')
displacement_print = Function(V_u, name='Displacement')
Temperature_print = Function(V_T, name='Delta_T')


# Split system functions to access components
u, T = split(state) #u.split(True) for postprocessing 
u_n, T_n = split(state_n) #u_n.split(True)for postprocessing

# Gradient: 
grad_w_T = grad(w_T)
grad_T = grad(T) 
grad_w_u = grad(w_u)

#------------------------------------------------
#          Define Boundary and Loading condition
#------------------------------------------------

dof_coordinates = V_u.tabulate_dof_coordinates()                    
dof_coordinates.resize((V_u.dim(), mesh.geometry().dim()))                           
dof_x = dof_coordinates[:, 0] 
dof_y = dof_coordinates[:, 1]
dof_z = dof_coordinates[:, 2]

#x-(horizontal), y-(vertical) 
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = dof_x.min())
right =  CompiledSubDomain("near(x[0], side) && on_boundary", side = dof_x.max())
bottom =  CompiledSubDomain("near(x[1], side) && on_boundary", side = dof_y.min())
top =  CompiledSubDomain("near(x[1], side) && on_boundary", side = dof_y.max())
back =  CompiledSubDomain("near(x[2], side) && on_boundary", side = dof_z.min())
front =  CompiledSubDomain("near(x[2], side) && on_boundary", side = dof_z.max())
top_left_corner =  CompiledSubDomain("near(x[0], side1) && near(x[1], side2) && on_boundary", side1 = dof_x.min(), side2 = dof_y.max())

unit_y_normal = Constant((0.0, 1.0, 0.0))

tsteps = np.linspace(0,t_max,numTSteps)

## Monotonous tensile load
top_disp_list = [Max_y_disp*curr_t/t_max for curr_t in tsteps]
bottom_disp_list = [0. for curr_t in tsteps]


#------------------------------------------------
#            Useful functions
#------------------------------------------------

zero = Constant(0.0)
one = Constant(1.0)
zero_vector = Constant([0.0]*ndim)
zero_state = Function(V)
fa_lift.assign(zero_state,[interpolate(zero_vector, V_u), interpolate(zero, V_T)])

#------------------------------------------------
#            Constitutive Relationships
#------------------------------------------------


# g_d = pow(one-d,2)
# g_d_prime = 2*(one-d)

# Lame parameter
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
# mu = Constant(mu) # Shear modulus
# lmbda = Constant(2*mu*nu/((1 - 2*nu))) 
I = Identity(ndim)  
strain = sym(grad(u))
thermal_strain = alpha*T*I
elastic_strain = strain - thermal_strain
stress = lmbda * tr(elastic_strain) * I + 2 * mu * elastic_strain



#Calculation of tensile load
traction_top_surface = dot(stress, unit_y_normal)
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
top.mark(boundary_subdomains,1)
ds_top = Measure("ds", domain=mesh, subdomain_data=boundary_subdomains, subdomain_id=1, metadata={'quadrature_degree': q_degree})
tensile_load = dot(traction_top_surface , unit_y_normal) *ds_top


## Initial Condition = Start with undeformed virgin material
state.assign(zero_state)
state_n.assign(zero_state)

# Heat residual

# Constant
g = Constant(0.)

# Spatially dependent
# g = Expression('(x[0] - 0.05) *(x[0] + 0.05) * (x[1] - 0.05) *(x[1] - 0.05) ', degree=4)

# Field dependent
# g = 0.01*(100. *T - T*T)


R = (1/tau)*inner(T-T_n, w_T) + kappa*inner(grad_T, grad_w_T) - inner(g, w_T)
# Deformation residual
R += inner(stress, grad_w_u)

R = R*dx_
J = derivative(R, state)


file_hdf5 = HDF5File(MPI.comm_world, './result/thermoelasticity_result.h5', 'w')
file_hdf5.write(mesh,'/mesh')
file_result_xdmf = XDMFFile(MPI.comm_world, './result/thermoelasticity_result_display.xdmf')

file_result_xdmf.parameters.update(
{
    "functions_share_mesh": True,
    "rewrite_function_mesh": False
})


frame_name = '/state_%s'%(str(1))
file_hdf5.write(state,frame_name) 
load_curve = np.zeros((numTSteps,2))
for t_index in range(1, numTSteps):
    print(f'------------t={tsteps[t_index]}------------')
    tau.assign(tsteps[t_index] - tsteps[t_index-1])
    #Get BC for current time step
    curr_top_disp = top_disp_list[t_index]
    curr_bottom_disp = bottom_disp_list[t_index]
    BC_u_top = DirichletBC(V.sub(0).sub(1), Constant(curr_top_disp), top)
    BC_u_bottom = DirichletBC(V.sub(0).sub(1), Constant(curr_bottom_disp), bottom)
    BC_u_left = DirichletBC(V.sub(0).sub(0), Constant(0), left)
    BC_T_left = DirichletBC(V.sub(1), Constant(0.), left)
    BC_T_right = DirichletBC(V.sub(1), Constant(100.), right)
    #BC_u_top_left = DirichletBC(V.sub(0).sub(0), Constant(0) ,top_left_corner, method="pointwise")

    bcs = [BC_u_top, BC_u_bottom, BC_u_left, BC_T_left, BC_T_right]

    problem = NonlinearVariationalProblem(R, state, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1E-8
    prm['newton_solver']['relative_tolerance'] = 1E-9
    prm['newton_solver']['maximum_iterations'] = 25
    #prm['newton_solver']["error_on_nonconvergence"] = False
    # prm['nonlinear_solver']='snes'
    # prm['snes_solver']['line_search']='bt'
    # prm['snes_solver']['linear_solver']='lu'
    # prm['snes_solver']['maximum_iterations'] = 500
    #prm['snes_solver']["error_on_nonconvergence"] = False
    solver.solve()
    state_n.assign(state)
    load_curve[t_index, :] = [curr_top_disp/0.1, assemble(tensile_load)*1e6] #Hard coded strain for current geometry
    print(f'Nominal Strain (unitless): {load_curve[t_index, 0]}, Tensile Load (kN) :{load_curve[t_index, 1]} ')   
    stress_print.assign(project(stress, V_stress, solver_type='cg'))
    displacement_print.assign(project(u, V_u, solver_type='cg'))    
    Temperature_print.assign(project(T, V_T, solver_type='cg'))    
    
    xdmf_index = t_index + 1

    file_result_xdmf.write(displacement_print,xdmf_index)
    file_result_xdmf.write(Temperature_print,xdmf_index)
    file_result_xdmf.write(stress_print,xdmf_index)
    
    frame_name = '/state_%s'%(str(t_index+1))
    file_hdf5.write(state,frame_name)  
file_result_xdmf.close()
file_hdf5.close()
np.savetxt('./result/load_curve.txt', load_curve, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)



