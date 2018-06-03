# RadReportAnnotator
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/aisinai/rad-report-annotator/master)

Authors: jrzech, eko

Provides a library of methods for automatically inferring labels for a corpus of radiological reports given a set of manually-labeled data. These methods are described in our publication [Natural Language–based Machine Learning Models for the Annotation of Clinical Radiology Reports](https://doi.org/10.1148/radiol.2018171093).

## Getting Started:

Click on the `launch binder` button at the top of this `README` to launch a remote instance in your browser using [binder](https://mybinder.org/). This requires no local configuration and lets you get started immediately. To see a demo of the library on data from the [Indiana University Chest X-ray Dataset (Demner-Fushman et al.)](https://www.ncbi.nlm.nih.gov/pubmed/26133894), please open `Demo Notebook.ipynb` and run all cells.

To configure your own local instance (assumes [Anaconda is installed](https://www.anaconda.com/download/)):

```
git clone https://www.github.com/aisinai/rad-report-annotator.git
cd rad-report-annotator
conda env create -f environment.yml
source activate rad_env
```
