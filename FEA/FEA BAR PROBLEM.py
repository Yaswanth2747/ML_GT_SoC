# This is a programme to run FEM for an axial loaded bar problem.
# by Yaswanth Ram Kumar
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

L = float(input("Enter length of the beam : "))
A = float(input("Enter the Area of cross section at the fixed end A(0) : "))
P = float(input("Enter the point load at x = L : "))
rho = float(input("Enter the density (rho) of the material : "))
E = float(input("Enter Young's Modulus (E) : "))  #enter values of the order 10e+10 for realistic results.
num_elements = int(input("Enter the number of elements to discretize 10: "))
num_nodes = num_elements+1
g = 9.8

def area(x):
    return A-(A/(2*L))*x

def element_stiffness(A1, A2):
    k = np.array([[1, -1], [-1, 1]])*(E*A1*A2)/(A1+A2)/L #derived from weak form equation through variational method, also available online. 
    return k

K_global = np.zeros((num_nodes, num_nodes))

for i in range(num_elements): #generating the global stiffness matrix
    A1 = area((i*L)/num_elements)
    A2 = area((i+1)*L/num_elements)
    k_element = element_stiffness(A1, A2)
    K_global[i:i+2, i:i+2] += k_element #carefully constructing global from element matrices.

K_reduced = np.delete(np.delete(K_global, 0, axis=0), 0, axis=1) #applying the Essential Boundary condition adn reducing the golbal matrix.
P_reduced = np.zeros(num_nodes-1)
P_reduced[-1] = P #taking the point load at x=L as the forcing vector's last element (Natural Boundary condition)

displacements = np.linalg.solve(K_reduced, P_reduced)
print("Displacements :\n", displacements)

reactions = np.dot(K_global, np.concatenate(([0], displacements)))-np.concatenate(([0], P_reduced))
print("Reactions :\n", reactions)

x_coords = np.linspace(0, L, num_nodes)
x_diff = np.diff(x_coords)
y_coords = np.zeros(num_nodes)
y_coords[1:] = displacements

strains = np.zeros(num_elements)
for i in range(num_elements):  #strain = du/dx
    strain = (y_coords[i+1]-y_coords[i])/x_diff[i] 
    strains[i] = strain

stresses = []   
for i in range(num_elements):  #stress = Young's Modulus x strain
    stress = E*strains[i]
    stresses.append(stress)
print("Stresses :\n", stresses)



fig = go.Figure()

#fig.add_trace(go.Scatter(x=x_coords[:-1], y=y_coords[1:], mode='markers+lines', marker=dict(color='blue'), name='Deformed Beam'))
fig.add_trace(go.Scatter(x=x_coords[:-1], y=y_coords[1:], mode='markers', marker=dict(color=strains, colorscale='viridis', colorbar=dict(title='Strain')), name='Strain'))

fig.update_layout(
    title='Deformed Beam with Color Representation of Strain',
    xaxis_title='Beam Length (m)',
    yaxis_title='Displacement (m)',
)

fig.show()
'''plt.figure(figsize=(10, 6))
plt.plot(x_coords[:-1], y_coords[1:], '-bo', label='Deformed Beam')
plt.xlabel('Beam Length (m)')
plt.ylabel('Displacement (m)')
plt.title('Deformed Beam with Color Representation of Strain')
plt.scatter(x_coords[:-1], y_coords[1:], c=strains, cmap='viridis', label='Strain')
plt.colorbar(label='Strain')
plt.show()'''