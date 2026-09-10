# %%
#Script only works when connected to Donders VPN, otherwise the paths cannot be found.
#Use python==3.8.10

from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import cylinder_models, sphere_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core import modeling_framework
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from os.path import join
import numpy as np
import dipy
import matplotlib.pyplot as plt
import scipy
from scipy.io import savemat
import numpy as np
import numba
import pathos
import os

# %%
os.chdir(r'/project/4180000.74/wessel/pilot/data/pilot1/analysis/')
os.listdir()

# %%
bvalues = np.loadtxt('bvals_virt.bval')  # given in s/mm^2
bvalues_SI = bvalues * 1e6  # now given in SI units as s/m^2
    
gradient_directions = np.loadtxt('pilot1_bvecs_merged.bvec')  # on the unit sphere
gradient_directions = np.transpose(gradient_directions)

delta = np.loadtxt('small_delta.txt')
delta = delta / 1000
Delta = np.loadtxt('big_delta.txt')
Delta = Delta / 1000

acq_scheme = acquisition_scheme_from_bvalues(bvalues_SI, gradient_directions, delta, Delta)

acq_scheme.print_acquisition_info


# %%
data = dipy.data.fetcher.load_nifti_data("slice_0010.nii.gz")
mask_file = dipy.data.fetcher.load_nifti_data("mask_0010.nii.gz")

# %%
# C1Stick
# lambda_par = isotropic diffusivity in m^2/s
#astrosticks = cylinder_models.C1Stick(mu=None,lambda_par=None)
astrosticks = cylinder_models.C1Stick()

# Spheres
#diameter=None, diffusion_constant=1.7e-09
sphere_small = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=1.0e-09)
#sphere_small = sphere_models.S2SphereStejskalTannerApproximation()
sphere_large = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=1.0e-09)
#sphere_large = sphere_models.S2SphereStejskalTannerApproximation()


# Hindered
#hindered = gaussian_models.G2Zeppelin(mu=None, lambda_par=None, lambda_perp=None)
hindered = gaussian_models.G2Zeppelin()

# Free
# lambda_iso = isotropic diffusivity in m^2/s -> FW 3 x10-3 mm2/s
free_water_ball = gaussian_models.G1Ball(lambda_iso= 3.0e-09)

# Configure the model with a Watson distribution for fiber orientation dispersion
watson_distribution = SD1WatsonDistributed(models=[astrosticks, hindered])
watson_distribution.set_fixed_parameter('C1Stick_1_lambda_par', 1.0e-9)

# `mu` represents the main orientation in spherical coordinates (theta, phi)
# `kappa` is the concentration parameter of the Watson distribution, related to dispersion

# %%
# Set up compartments incl. dispersed bundle
model = MultiCompartmentModel(models=[sphere_small, sphere_large, watson_distribution, free_water_ball])
model.set_fixed_parameter('G1Ball_1_lambda_iso', 3.0e-09)
#model.set_parameter_optimization_bounds('G2Zeppelin_1_lambda_par', [0.1e-09, 2.9e-09])
#model.set_parameter_optimization_bounds('G2Zeppelin_1_lambda_perp', [0.1e-09, 2.9e-09])
model.set_parameter_optimization_bounds('S4SphereGaussianPhaseApproximation_1_diameter', [0.1e-06, 12.0e-06])
model.set_parameter_optimization_bounds('S4SphereGaussianPhaseApproximation_2_diameter', [12.1e-06, 24.0e-06])
#model.set_parameter_optimization_bounds('S2SphereStejskalTannerApproximation_1_diameter', [0.1e-06, 12.0e-06])
#model.set_parameter_optimization_bounds('S2SphereStejskalTannerApproximation_2_diameter', [12.1e-06, 24.0e-06])
#Can also restrict fractions:
#model.set_parameter_optimization_bounds('partial_volume_2', [0.1, 1])
#model.set_parameter_optimization_bounds('partial_volume_3', [0.1, 1])
#model.set_parameter_optimization_bounds('partial_volume_0', [0.01, 0.50])  
#model.set_parameter_optimization_bounds('SD1WatsonDistributed_1_SD1Watson_1_odi', [0.01, 0.8])  

# %%
print(data.shape)
print(mask_file.shape)

# %%
plt.figure(figsize=(4, 4))
plt.imshow(data[:,0,:,0], aspect='auto')
plt.title('b0 of brain data')
plt.axis('off');
plt.savefig('b0_image.png', dpi=300, bbox_inches='tight')

# %%
model.parameter_cardinality

# %%
data=np.squeeze(data)
mask_file=np.squeeze(mask_file)
print(data)

# %%
microg_fit = model.fit(
    acq_scheme, data, mask=mask_file,
    solver='mix', maxiter=100,
    use_parallel_processing=True, number_of_processors=16 #make sure is the same as requested in the HPC job submission script
)

# %% [markdown]
microg_fit.fitted_parameters

fitted_parameters = microg_fit.fitted_parameters
import pickle
with open ('fitted_parameters.pkl', 'wb+') as f:
    pickle.dump(fitted_parameters, f)


fig, axs = plt.subplots(3, 4, figsize=[15, 10])
axs = axs.ravel()

counter = 0
for name, values in fitted_parameters.items():
    if values.squeeze().ndim != 2:
        continue
    #cf = axs[counter].imshow(values.squeeze().T, origin=True, interpolation='nearest')#
    cf = axs[counter].imshow(values.squeeze().T, origin='lower',interpolation='nearest')
    axs[counter].set_title(name)
    fig.colorbar(cf, ax=axs[counter], shrink=0.5)
    counter += 1

plt.savefig('fitted_parameters.png', dpi=300, bbox_inches='tight')

# %%
fitted_parameters
scipy.io.savemat('test.mat', fitted_parameters)
