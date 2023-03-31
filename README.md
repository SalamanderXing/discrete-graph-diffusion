# Discrete Graph Diffusion

## Project Structure

- `pytorch` $\rightarrow$ Almost identical to the [original code](https://github.com/cvignac/DiGress), adapted for debugging.
- `discrete_graph_diffusion` $\rightarrow$ new code in JAX

## Status

Only the graph transformer works.

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
Navigate into the `discrete_graph_diffusion` directory and run `mate summary`. If everything worked well you should see a nice representation of the project. 

Then you can run `mate run graph_transformer_test` to run a simple test that checks the graph transformer model is working.

---

<small>builtwithmate</small>
