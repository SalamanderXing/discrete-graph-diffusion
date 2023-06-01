# Discrete Graph Diffusion

## Project Structure

- `pytorch` $\rightarrow$ Almost identical to the [original code](https://github.com/cvignac/DiGress), adapted for debugging.
- `graph_diffusion` $\rightarrow$ new code in JAX

### The `graph_diffusion` folder

All the code regarding diffusion lies in the `trainers/ddd_trainer` folder. 
The entrypoint is the file `trainers/ddd_trainer/discrete_denoising_diffusion.py`

## Status

- Graph transformer: looks like its working
- Data loader: I kept the same from the original code.
- Trainer (diffusion stuff): only training seems to work.

## Running the code

The project is created with Mate, which is just a way to run a project organized into isolated python modules.

#### Install Mate

Somewhere on your machine:
```
git clone https://github.com/salamanderxing/mate
cd mate
pip install -e .
```

#### Running the code
Navigate into the `graph_diffusion` directory and run `mate summary`. If everything worked well you should see a nice representation of the project. 

Then you can run `mate run ddd_tr_QM9 train` to run the main experiment.

---

<small>builtwithmate</small>
