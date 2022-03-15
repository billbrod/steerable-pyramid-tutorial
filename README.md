# steerable-pyramid-tutorial

Brief introduction to steerable pyramids. 

Note that this tutorial uses the implementation of the steerable pyramids found
in [pyrtools](https://github.com/LabForComputationalVision/pyrtools/), which is
built using numpy. For most uses, however, you probably want the version
implemented in
[plenoptic](https://github.com/LabForComputationalVision/plenoptic/), which is
built using `pytorch` and thus has automatic differentiation (as well as being
GPU-compliant and thus, faster). The use is slightly different, but the ideas
covered here should serve you well (you can see [plenoptic's
tutorial](https://plenoptic.readthedocs.io/en/latest/tutorials/03_Steerable_Pyramid.html)
for details on how to use its steerable pyramid implementation).

# Requirements

See `requirements.txt` (you can install them with `pip install -r
requirements.txt`). You'll also need
[jupyter](https://jupyterlab.readthedocs.io/en/stable/) if you wish to view the
notebook

# Further reading

- Brian Wandell's [Foundations of
  Vision](https://foundationsofvision.stanford.edu/chapter-8-multiresolution-image-representations/),
  chapter 8 (the rest of the book is helpful if you want to
  understand the basics of the visual system).
- [Adelson et al, 1984, "Pyramid methods in image
  processing".](http://persci.mit.edu/pub_pdfs/RCA84.pdf)
- [Freeman and Adelson, 1991, "The Desig and Use of Steerable
  Filters"](https://people.csail.mit.edu/billf/www/papers/steerpaper91FreemanAdelson.pdf)
- [Simoncelli and Freeman, 1995, "The Steerable Pyramid: A Flexible Architecture
  for Multi-Scale Derivative
  Computation"](https://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf)
- Notes from David Heeger on [steerable
  filters](http://www.cns.nyu.edu/~david/handouts/steerable.pdf)
- Notes from Eero Simoncelli on [the Steerable
  Pyramid](http://www.cns.nyu.edu/~eero/STEERPYR/)
- Lectures and course notes from the NYU [Mathematical Tools for Data Science
  course](https://cds.nyu.edu/math-tools/), the "Beyond Fourier"
  lectures.
