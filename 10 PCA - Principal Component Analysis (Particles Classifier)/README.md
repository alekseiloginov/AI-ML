PCA - Principal Component Analysis (Particles Classifier)

The data set was generated by a Monte Carlo program, Corsika, 
described in D. Heck et al., CORSIKA, 
A Monte Carlo code to simulate extensive air showers, 
Forschungszentrum Karlsruhe FZKA 6019 (1998).

Goals:
- Classify particles into gamma (signal) or hadrons (background).
- Perform PCA to get a new set of features, and select the features that contain the most information.
- Use `numpy` and `sklearn` libraries for that.