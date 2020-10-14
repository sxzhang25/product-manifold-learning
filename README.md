# Product Manifold Learning
Code for reproducing the results from the paper <b>Product Manifold Learning</b>.

## Dependencies
Before running this project, make sure to `pip install` the following packages:

* `cvxpy`
* `scipy`
* `scikit-learn`
* `numpy`

This project was developed using Python 3.7.0.

## Running experiments
To run the algorithm on a geometric dataset, use `run_synthetic.py` on the appropriate parameters `.json` file.

For example, to run the algorithm on data sampled from a 2d rectangle with 3d noise, run

```
python run_synthetic.py params_rectangle3d.json
```

To run the algorithm on synthetic cryo-EM data, use `run_cryo_em.py` on the appropriate parameters `.json` file.

```
python run_cryo_em.py params_cryo_em.json
```

The plots will be produced if `generate_plots` is set to `True` in the files `run_*.py`.

To set the directory for saving figures, change the `image_dir` variable in `run_synthetic.py` or `run_cryo_em.py`.

## Generating data
To generate new data for the algorithm, set the `precomputed` field in the `.json` parameters file to `false`.

To set the directory for saved datasets, change the `data_dir` variable in `run_synthetic.py` or `run_cryo_em.py`.

## Customizing algorithm parameters
The settings for the algorithm used in the experiments from the paper are stored in a `.json` file.

The format of parameters for geometric data is:

* `test_name`: A user-defined name for the experiment, used for naming saved figures.
* `precomputed`: If `true`, data will be generated before running the algorithm. Otherwise, pre-generated data will be used.
* `dimensions`: The dimensions of the data manifold.
* `noise`: The amount of Gaussian noise to add, in the range [0, 1].
* `n_samples`: The number of samples in the dataset. If `precomputed` is set to false, the generated dataset will contain this many samples.
* `seed`: A random seed.
* `datatype`: The type of manifold. See the `generate_synthetic_data()` method in `synthetic.py` to see possible data types.
* `sigma`: The width of the kernel for constructing the data graph.
* `n_comps`: The desired number of factors to extract.
* `n_eigenvectors`: The number of eigenvectors to compute.
* `lambda_thresh`: The threshold for the eigenvalue criterion (see Section 3.1 of the paper for more details).
* `corr_thresh`: The threshold for the correlation criterion (see Section 3.1 of the paper for more details).
* `K`: The number of votes needed to pass the caucusing step (see Section 3.2 of the paper for more details).

The format of parameters for cryo-EM data is:
* `test_name`: A user-defined name for the experiment, used for naming saved figures.
* `var`: The variance of the Gaussian noise to add to the images.
* `precomputed`: If `true`, data will be generated before running the algorithm. Otherwise, pre-generated data will be used.
* `n_samples`: The number of samples in the dataset. If `precomputed` is set to false, the generated dataset will contain this many samples.
* `seed`: A random seed.
* `x`: The range of motion for the stretch component in the x-direction.
* `y`: The range of motion for the stretch component in the y-direction.
* `sigma`: The width of the kernel for constructing the data graph.
* `n_comps`: The desired number of factors to extract.
* `n_eigenvectors`: The number of eigenvectors to compute.
* `lambda_thresh`: The threshold for the eigenvalue criterion (see Section 3.1 of the paper for more details).
* `corr_thresh`: The threshold for the correlation criterion (see Section 3.1 of the paper for more details).
* `K`: The number of votes needed to pass the caucusing step (see Section 3.2 of the paper for more details).
