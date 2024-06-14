# fenicsx_examples
New examples for solving mechanical problems using fenicsx. 
legacy_* codes use the old version of fenics available with following docker image: quay.io/fenicsproject/stable:current

# Docker usage instructions

Step 1: Install Docker

Step 2: Fix the image to be pulled 
```
export image_name=dolfinx/dolfinx
export tag=stable
```

Step 3: Pull the image to your local system

```
docker pull  $image_name:$tag
```

Step 4: Run a container named fenicsx from the image on your local system
```
docker run --name fenicsx -dit -w /home/fenics/shared -v $(pwd):/home/fenics/shared $image_name:$tag
```

Step 5: Go into the container
```
docker exec -it fenicsx bash
```
Coming out of container: `ctrl+D`

Step 6: Run the test case
```
python3 linearelasticity.py
```
Other useful commands: 

1. Removing an image (You will not need this unless you used the wrong image)
```
docker rmi $image_name:$tag
```

2. Removing a container named fenicsx
```
docker rm fenicsx
```

3. Checking status of all containers
```
docker ps -a
```

4. Checking status of all images
```
docker ps -i
```

5. Running with privileges to get mount on
```
docker run --name fenicsx -dit --privileged -w /home/fenics/shared -v $(pwd):/home/fenics/shared $image_name:$tag
```

# linearelasticity.py

#### Imports 
```
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np

```
fenics uses multiple libraries: 
`dolfinx.mesh`: Provides support for meshing
`dolfinx.fem`: Provides support for various Finite Element functionality
`dolfinx.plot`: Provides support for plotting (I do not use it)
`dolfinx.io`: Provides support for input/output. We will use it to import/export mesh and solution etc. 
`dolfinx.default_scalar_type`:  Use to set the field type
`dolfinx.fem.petsc`: petsc library offer various large-scale numerical solvers (https://petsc.org/release/overview/nutshell/)
`mpi4py`: Provides support for parallel programming
`ufl`: Unified form language (support for computation on variational forms)

#### Mesh, elements,  domain and boundaries
```
domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],
[20, 6, 6], cell_type=mesh.CellType.hexahedron)

```


```
V = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
```

This step for defining $\partial \Omega_D$
```
def clamped_boundary(x):
	return np.isclose(x[0], 0)
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

```

We can run with other elements as well:
```
domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],
[20, 6, 6], cell_type=mesh.CellType.tetrahedon)
```

#### Setting boundary conditions
```  
#Dirichlet boundary conditions
u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

#Neumann boundary conditions
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

#Boundary integral
```


#### Setting Weak form
```
#Test and trial functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
```
`u` is a solution from the vector space `V` (finite element solution)
`v`represents arbitrary perturbations in the solution `u` (weighting function)
```
#Function definition for stress and strain
def epsilon(u):
	return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma(u):
	return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)
```
Notice the `ufl` functions: 
`ufl.sym`: symmetric part of the matrix
`ufl.grad(u)`: gradient of   
`ufl.Identity(len(u))`: Identity matrix
`ufl.nabla_div(u)`: divergence of the vector field `u`
$$
\epsilon = \frac{1}{2} \left(\begin{bmatrix}\frac{\partial u_1}{\partial x_1}& \frac{\partial u_1}{\partial x_2} & \frac{\partial u_1}{\partial x_3}\\
\frac{\partial u_2}{\partial x_1}& \frac{\partial u_2}{\partial x_2} & \frac{\partial u_2}{\partial x_3}\\
\frac{\partial u_3}{\partial x_1}& \frac{\partial u_3}{\partial x_2} & \frac{\partial u_3}{\partial x_3}\\
\end{bmatrix} + \begin{bmatrix}\frac{\partial u_1}{\partial x_1}& \frac{\partial u_2}{\partial x_1} & \frac{\partial u_3}{\partial x_1}\\
\frac{\partial u_1}{\partial x_2}& \frac{\partial u_2}{\partial x_2} & \frac{\partial u_3}{\partial x_2}\\
\frac{\partial u_1}{\partial x_3}& \frac{\partial u_2}{\partial x_3} & \frac{\partial u_3}{\partial x_3}\\
\end{bmatrix}\right)
$$

$$
\sigma = \lambda (\epsilon_{11} + \epsilon_{22} + \epsilon_{33}) \mathbb{I}+2\mu \epsilon  
$$
Define a differential form for the boundary integral
```
# Define measure
ds = ufl.Measure("ds", domain=domain)  

```



```
f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

```
$inner(a,b) = \int (a\cdot b) dx$ when $a$ and $b$ are vectors

#### Solver
```
#Solver
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

uh = problem.solve()

```
Here, `a` and `L` are "like" to `K` and `F` respectively. `LinearProblem` class reduces the system by applying the Dirichlet BCs. 
Resource for solver information: https://petsc.org/release/overview/nutshell/
$u^h$
#### Paraview plotting
```
#Paraview plotting

with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
xdmf.write_mesh(domain)
uh.name = "Deformation"
xdmf.write_function(uh)
```

