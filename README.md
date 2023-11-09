# Discrete Graph Diffusion

DDGD, discrete graph diffusion is a diffusion model that is able to generate
graphs and uses a discrete categorical distribution instead of a gaussian to add
noise to the graph. While this project was tested for graph generation, its
primary focus is to get a good value for the Evidence Lower BOund (ELBO). A good
(low) ELBO value value roughly indicatest that the model has modeled well the
underlying graph distribution.

$$-log(p(\mathbf{x})) \leq \text{ELBO}(\mathbf{x})$$

This model was tested on the QM9 dataset.

#### Install Mate

Somewhere on your machine:

```
git clone https://github.com/salamanderxing/mate
cd mate
pip install -e .
```

#### Running the code

Navigate into the `graph_diffusion` directory and run `mate summary`. If
everything worked well you should see a nice representation of the project.

Then you can run `mate run ddd_tr_QM9 train` to run the main experiment.

---

<small>builtwithmate</small>
