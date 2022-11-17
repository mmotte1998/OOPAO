# OOPAO
Object Oriented Python Adaptive Optics (OOPAO) is a project under development to propose a python-based tool to perform end-to-end AO simulations.
This code is inspired from the OOMAO architecture: https://github.com/cmcorreia/LAM-Public developped by C. Correia and R. Conan (https://doi.org/10.1117/12.2054470). 
The project was initially intended for personal use in the frame of an ESO project. It is now open to any interested user. 

## FUNCTIONALITIES

	_ Atmosphere: 		Multi-layers with infinitely and non-stationary phase screens, conditions can be updated on the fly if required
	_ Telescope: 		Default circular pupil or user defined, with/without spiders
	_ Deformable Mirror:	Gaussian Influence Functions (default) or user defined, cartesian coordinates (default) or user defined
	_ WFS: 			Pyramid, SH-WFS (diffractive and geometric)
	_ Source: 		NGS or LGS
	_ Control Basis: 	KL modal basis, Zernike Polynomials

## LICENSE
This project is licensed under the terms of the MIT license.

## INSTALLATION 

### (Recommended) Creating a virtual environment

It is always recommended that you use a virtual environment. First create it:

```
python -m venv venv

# or

python3 -m venv venv

```

And finally activate it:

```
# Unix
source ./venv/bin/activate

# or

# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

After the environment is set up and activated, this package can then be easily installed. Anytime you wish to use this
package, you should activate the respective environment.

### Using `pip`

First clone the repository:

```
https://github.com/cheritier/OOPAO.git
```

And then install the package using `pip`:

```
python -m pip install -e OOPAO

# or 

python3 -m pip install -e OOPAO
```


## MODULES REQUIRED
The code is written for Python 3 (version 3.8.8) and requires the following modules:
```
joblib          => paralleling computing
scikit-image    => 2D interpolations
astropy         => handling of fits files
pyFFTW          => optimization of the FFT  
mpmath          => arithmetic with arbitrary precision
jsonpickle      => json files encoding
aotools         => zernike modes and functionalities for atmosphere computation
numba           => required in aotools
```
If GPU computation is available:
```
cupy => GPU computation of the PWFS code (Not required)
```
## CODE OPTIMIZATION

OOPAO multi-threading is based on the use of the numpy package built with the mkl library, make sure that the proper numpy package is installed to make sure that the operations are multi-threaded. 
To do this you can use the numpy.show_config() function in your python session: 

import numpy
numpy.show_config()


## CONTRIBUTORS
C.T. Heritier, C. Vérinaud

## ACKNOWLEDGEMENTS
This tool has been developped during the Engineering & Research Technology Fellowship of C. Héritier funded by ESO. 
Some functionalities of the code make use of the aotools package developped by M. J. Townson et al (2019). See https://doi.org/10.1364/OE.27.031316.