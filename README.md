# Image Inpainting with Global Gradient Statistics

This is a python script to fill holes in grayscale images.

The command line arguments are: filename, x1, x2, y1, y2, T.

(x1, y1) is the upper left coordinate of the hole, and (x2, y2) is the lower
right coordinate of the hole.

T is the number of iteratons in the simulated annealing step.

Example: python inpaint.py circle_bw_lowres.jpg 10 16 11 25 10000 

How the algorithm works:
* First, collect global statistics about gradient magnitude and gradient
  angles. For example, a square has either 90 degree gradient angle or 0 degree
  gradient angle, whereas a circle has uniform gradient angle.
* Construct a maximum likelihood estimator based on gradient statistics.
* Solve the MLE with simulated annealing.

See report.pdf for more details.
