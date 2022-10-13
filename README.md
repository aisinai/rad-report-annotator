# RadReportAnnotator

Authors: jrzech, eko

Provides a library of methods for automatically inferring labels for a corpus of radiological reports given a set of manually-labeled data. These methods are described in our publication [Natural Language–based Machine Learning Models for the Annotation of Clinical Radiology Reports](https://doi.org/10.1148/radiol.2018171093).

## Getting Started:

To configure your own local instance (assumes [Anaconda is installed](https://www.anaconda.com/download/)):

```
git clone https://www.github.com/aisinai/rad-report-annotator.git
cd rad-report-annotator
conda env create -f environment.yml
source activate rad_env
python -m ipykernel install --user --name rad_env --display-name "Python (rad_env)"
```

*Note as of Oct 11, 2022: this conda environment builds on Linux and Windows, but not on Mac as older versions of gensim for Mac are not available in conda-forge.* 

To see a demo of the library on data from the [Indiana University Chest X-ray Dataset (Demner-Fushman et al.)](https://www.ncbi.nlm.nih.gov/pubmed/26133894), please open `Demo Notebook.ipynb` and run all cells.

