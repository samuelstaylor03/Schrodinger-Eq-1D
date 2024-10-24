'''
1-Dimensional Time-Independent Schrodinger Equation Solver

Note: This is purely an educational exercise to understand how a real-space grid can be used to
      solve the Schroinger Equation with different potentials.

'''

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh



class SchrodingerSolver:
    def __init__(self, units="SI"):
        self.units = units[0].strip().lower()
        if self.units == "s":
            self.h = 6.62607015e-34 # J*s = Joule*second = (Joule * hertz^(-1)) ... from E = hf (energy of a photon)
            self.hbar = self.h / (2*math.pi) # Planck's reduced constant
            self.electron_mass = 9.1093837e-31 # Electron mass in kilograms
        elif self.units == "e":
            self.hbar = 0.658211899 # Planck constant over 2*pi in eV*fs
            self.h2m = 3.80998174348 # (hbar^2)/(2*m_e) in eV*Angstom^2
            self.electron_mass = (2*(self.h2m)/ (self.hbar**2))
        elif self.units == "a":
            self.hbar = 1.0
            self.h = self.hbar * 2 * math.pi
            self.electron_mass = 1
        else:
            print("Invalid units: unable to initialize.")
            
        
    def boot_and_input(self,user_input=False, potential_num=1):
        print("/// Welcome to Schrodinger Equation Solver:")
        if user_input:
            print("""List of Potentials:
                \n  1) Simple Harmonic Oscillator
                \n  2) Infinite Square Well 
                \n  3) Finite Square Well
                \n  4) Step Potential
                \n  5) Double Finite Square Well
                \n  5) Other\n""")
            self.potential_num = int(input("Choose your potential (enter the number): "))
        else: 
            self.potential_num = potential_num
        
    
    def initialize_grid(self,user_input=False):
        print("\n/// USING REAL-SPACE GRID TO SOLVE\n")
        if user_input:
            self.min_grid_val = int(input("Enter the minimum grid value: "))
            self.max_grid_val = int(input("Enter the maximum grid value: "))
            if self.max_grid_val < self.min_grid_val:
                raise Exception("ERROR: INVALID GRID SIZE")
            self.number_steps = int(input("Enter the number of steps to be used the grid: "))
        else:
            self.min_grid_val = -5
            self.max_grid_val = 5
            self.number_steps = 100
        self.grid_spacing = (self.max_grid_val - self.min_grid_val) / self.number_steps # (b-a)/n
        self.grid = [] # 1-D array of x-values that represent the grid to solve for psi on
        grid_point = self.min_grid_val
        while grid_point <= self.max_grid_val:
            self.grid.append(grid_point)
            grid_point += self.grid_spacing       
        self.number_points = len(self.grid)

    
    def init_hamiltonian_operator(self):
        kinetic_energy_operator = self.init_kinetic_energy()
        potential_energy_operator = self.init_potential_energy()
        self.hamiltonian_operator = self.kinetic_energy_matrix + self.potential_energy_matrix
        self.overlap_matrix = np.identity(self.number_points)
        

    """Initialize a kinetic energy matrix using 9-point finite difference.
            *  Second Derivative 8-order accuracy (9-point) look at 9 different points
       
                -4      -3    -2    -1    0    1    2     3      4
            (-1/560	8/315	-1/5	8/5	-205/72	8/5	-1/5	8/315	-1/560)
    """     
    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
    def init_kinetic_energy(self):
        #                                    0      1     2     3       4
        finite_difference_coefficients = [-205/72, 8/5, -1/5, 8/315, -1/560] #initialize finite difference coefficients
        
        # Diagonally fill the array with finite diff. coefficients
        array_2d =[[0 for _ in range(self.number_points)] for _ in range(self.number_points)]  #initialize 2-D matrix with all 0s
        for i in range(len(array_2d )):
            for j in range(len(array_2d )):
                if abs(i-j) < len(finite_difference_coefficients):
                    array_2d [i][j] = finite_difference_coefficients[abs(i-j)]
        second_deriv_matrix = np.matrix(array_2d) / (self.grid_spacing**2)
        # Kinetic energy operator. -hbar/2m * second_deriv
        self.kinetic_energy_matrix = (-1 * self.hbar) / (2*self.electron_mass) * second_deriv_matrix 
        
    
    def init_potential_energy(self):
        array_2d = [[0 for _ in range(self.number_points)] for _ in range(self.number_points)]  #initialize 2-D matrix with all 0s
        self.potential_energy_matrix = np.matrix(array_2d, dtype='float64')
        if int(str(self.potential_num)[0]) == 1:
            self.harmonic_oscillator()
            self.potential_num=self.potential_num % 10
        if int(str(self.potential_num)[0]) == 2:
            self.infinite_square_well()
            self.potential_num=self.potential_num % 10
        if int(str(self.potential_num)[0]) == 3:
            self.finite_square_well()
            self.potential_num=self.potential_num % 10
        if int(str(self.potential_num)[0]) == 4:
            self.step_potential()
            self.potential_num=self.potential_num % 10
        if int(str(self.potential_num)[0]) == 5:
            self.double_square_well()
            self.potential_num=self.potential_num % 10
        if int(str(self.potential_num)[0]) == 6:
            self.free_electron()
            self.potential_num=self.potential_num % 10
        else:
            print("TODO- ADD other")
        # Filling a 1-D array of the potential values. Fills diagonally from the potential energy matrix
        self.potential_array = [0 for _ in range(self.number_points)]
        for i in range(self.number_points):
            self.potential_array[i] = self.potential_energy_matrix[i, i]
    
    
    
    def harmonic_oscillator(self, input=False):
        omega = 1.0
        harmonic_oscillator_array = np.zeros(len(self.grid), dtype='float64')
        for i in range(self.number_points):
            x = self.grid[i]
            harmonic_oscillator_array[i] = 1/2 * self.electron_mass * omega**2 * x**2
        harmonic_oscillator_matrix = np.diag(harmonic_oscillator_array)
        self.potential_energy_matrix += harmonic_oscillator_matrix 
    
    
    def infinite_square_well(self, input=False, a=-1,b=1):
        large_value = 1e12  # A very large number to simulate infinity
        infinite_square_well_array = np.zeros(self.number_points, dtype='float64')
        
        for i in range(self.number_points):
            x = self.grid[i]
            if x < a:
                infinite_square_well_array[i] = large_value
            elif x > b:
                infinite_square_well_array[i] = large_value

        infinite_square_well_matrix = np.diag(infinite_square_well_array)
        self.potential_energy_matrix += infinite_square_well_matrix
    
    
    def finite_square_well(self, input=False, height_depth=-1, a=-1, b=1, y_level=0):
        finite_square_well_array = np.zeros(self.number_points, dtype='float64')
        for i in range(self.number_points):
            x = self.grid[i]
            if x < a:
                finite_square_well_array[i] = y_level
            elif x > a and x < b:
                finite_square_well_array[i] = height_depth
            elif x > b:
                finite_square_well_array[i] = y_level
        finite_square_well_matrix = np.diag(finite_square_well_array)
        self.potential_energy_matrix += finite_square_well_matrix
    
    
    def step_potential(self, input=False):
        x_jump = 0 # value where step jumps up at
        y_lower = 0 # values that are < x
        y_upper = 5 # values that are > x
        step_potential_array = [0 for _ in range(self.number_points)]
        for i in range(self.number_points):
            x = self.grid[i]
            if x < x_jump:
                step_potential_array[i] = y_lower
            else:
                step_potential_array[i] = y_upper
        step_potential_matrix = np.diag(step_potential_array)
        self.potential_energy_matrix += step_potential_matrix
           

    def double_square_well(self, input=False, left_well_a=-2, left_well_b=-1, left_well_height_depth=-1,
                                              right_well_a=1, right_well_b=2, right_well_height_depth=-1,
                                              y_level=0):
        double_square_well_array = np.zeros(self.number_points, dtype='float64')
        for i in range(self.number_points):
            x = self.grid[i]
            if x < left_well_a:
                double_square_well_array[i] = y_level
            elif x > left_well_a and x < left_well_b:
                double_square_well_array[i] = left_well_height_depth
            elif x > left_well_b and x < right_well_a:
                double_square_well_array[i] = y_level
            elif x > right_well_a and x < right_well_b:
                double_square_well_array[i] = right_well_height_depth
            elif x > right_well_b:
                double_square_well_array
        double_square_well_matrix = np.diag(double_square_well_array)
        self.potential_energy_matrix += double_square_well_matrix
        
        
    def free_electron(self):
        return
    
           
    def plot(self, n=3, SAVE=False, SHOW=True, PLOT_PROBABILITY=False):
        """Interpret eigensolutions"""
        for item in self.energy[:n]:
            print(item)
        
        fig, ax1 = plt.subplots()

        # Plotting wavefunctions on the primary y-axis
        if PLOT_PROBABILITY:
            self.psi *= self.psi
        
        for i in range(n):
            ax1.plot(self.grid, self.psi[:, i], label=f'Energy Level {i+1}', color='tab:blue', alpha=(1-(i*(1/n))))
        
        
        
        ax1.set_ylabel('$\psi(x)$')
        
        # Setting x-axis label based on units
        if self.units == "e":
            ax1.set_xlabel('$x$ [m]')
        elif self.units == "A":
            ax1.set_xlabel('$x$ [Angstrom]')
        elif self.units == "a":
            ax1.set_xlabel('$x$ [a.u.]')
        
        # Secondary y-axis for the potential energy
        ax2 = ax1.twinx()
        ax2.plot(self.grid, self.potential_array, label='Potential Energy', color='tab:red')
        ax2.set_ylabel('Potential Energy')

        # Set limits for the primary y-axis
        ax1.set_xlim(self.min_grid_val, self.max_grid_val)
        
        if abs(max(self.potential_array)) > abs(min(self.potential_array)):
            ax1.set_ylim(-abs(max(self.psi[:,0])*3), abs(max(self.psi[:,0])*3))
        else:
            ax1.set_ylim(-abs(min(self.psi[:,0])*3), abs(min(self.psi[:,0])*3))

        if self.potential_num==1 or self.potential_num==2:
            ax2.set_ylim(-abs(max(self.psi[:,0]*3)), max(self.psi[:,0]*3))
        elif abs(max(self.potential_array)) > abs(min(self.potential_array)):
            ax2.set_ylim(-abs(max(self.potential_array)*1.5), abs(max(self.potential_array)*1.5))
        else:
            ax2.set_ylim(-abs(min(self.potential_array)*1.5), abs(min(self.potential_array)*1.5))        

        # Legends for both y-axes
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        if SAVE:
            plt.savefig("wave_function.png", format='png')
        if SHOW:
            plt.show()
            
            
    def normalize(self, n):
        for i in range(n):
            # Calculate the integral of the square of psi using the trapezoidal rule
            integral = np.trapz(np.abs(self.psi[:, i])**2, self.grid)
            # Compute the normalization factor
            normalization_factor = np.sqrt(integral)
            # Normalize psi
            self.psi[:, i] /= normalization_factor
        return
        
    def check_normalization(self):
        sum=0
        self.psi_ground_state_squared = self.psi[:,0] * self.psi[:,0]
        for val in self.psi_ground_state_squared:
            sum += val * self.grid_spacing
        print("the integral of psi_squared is:", sum) # should roughly equal 1 if psi is normalized


    def solve(self):
        self.energy, self.psi = eigh(self.hamiltonian_operator, self.overlap_matrix)
        

    def initialize(self,user_input=False,potential_num=1):
        self.boot_and_input(user_input, potential_num)
        self.initialize_grid(user_input)
        self.init_hamiltonian_operator()
        self.solve()
    

def main():
    solver = SchrodingerSolver(units="a")
    solver.initialize(user_input=False,potential_num=1)
    solver.solve()
    solver.normalize(n=3)
    solver.check_normalization()
    solver.plot(n=3,SAVE=False,SHOW=True)    
   
    
if __name__ == "__main__":
    main()