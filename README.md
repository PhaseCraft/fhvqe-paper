# fhvqe
Variational Quantum Eigensolver for the Fermi-Hubbard model. Code developed by Phasecraft Ltd and corresponding to the paper

>Observing ground-state properties of the Fermi-Hubbard model using a scalable algorithm on a quantum computer.
>Stasja Stanisic, Jan Lukas Bosse, Filippo Maria Gambetta, Raul A. Santos, Wojciech Mruczkiewicz, Thomas E. O'Brien, Eric Ostby and Ashley Montanaro.
>arXiv:2112.02025


# Installation

Recommended: using anaconda or similar virtual environment software.
Requires:
- python (3.9.7)
- cirq (0.11.1)
- openfermion (1.1.0)
- numpy (1.21.2)
- gitpython
- pandas

Installation of cirq from github recommended.
```
python -m pip install git+https://github.com/quantumlib/Cirq.git
```

# Usage

The configuration files for various experiments can be found in the config directory.

The vqe_experiment module takes three command-line arguments. One tells it the location of the configuration file: -f / --config. The other gives whether the experiment is run as simulation (-s / --simulation), exact (-e / --exact) or on chip. These settings overwrite the setting in the configuration file.


Assuming we want to run vqe_experiment.py, with configuration file config.json, and simulated, a call would be something like:

```
python -m fhvqe.vqe_experiment -s -f config.json
```

Similarly for exact, a call would be something like:

```
python -m fhvqe.vqe_experiment -e -f config.json
```

There can be multiple configuration files given at the same time.

```
python -m fhvqe.vqe_experiment -e -f config1.json -f config2.json
```

Finally, there are various debug flags throughout the code that can be switched off when the experiment is running on the actual device (or generally, for performance purposes) by adding a call to -O.


```
python -O -m fhvqe.vqe_experiment -f config.json
```

