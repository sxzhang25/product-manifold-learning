# Product Manifold Learning
Code for reproducing the results from the paper <b>Product Manifold Learning</b>.

## Dependencies
Before running this project, make sure to `pip install` the following packages:

* `cvxpy`
* `scipy`
* `scikit-learn`
* `numpy`

This project was developed using Python 3.7.0.

## Generating data

To generate a dataset, create an appropriate .json file in the `params` folder. Then, run

```
python generate_data.py <path/to/params/file1> <path/to/params/file2> ...
```

Note that multiple datasets can be generated at once by passing a list of .json files. For example, 

```
python generate_data.py params/params_rectangle3d.json params/params_torus.json
```

will generate two separate datasets. The datasets will be .pickle files located in the `data/` folder.

Specifications for the datasets used in the paper can be changed by changing the keys in the params file.

The params files for geometric data are formatted as follows:

* `name`: A user-defined name for the experiment, used for naming saved figures.
* `dimensions`: The dimensions of the data manifold.
* `noise`: The amount of Gaussian noise to add, in the range [0, 1].
* `n_samples`: The number of samples in the dataset.
* `seed`: A random seed.
* `datatype`: The type of manifold. Refer to `generate_data.py` to see possible data types.

The params files for cryo-EM data are formatted as follows:

* `name`: A user-defined name for the experiment, used for naming saved figures.
* `var`: The variance of Gaussian noise to add to the images.
* `n_samples`: The number of samples in the dataset.
* `seed`: A random seed.
* `x_stretch`: The range [-x, x] which the stretching subunit can stretch in the x-direction.
* `y_stretch`: The range [-y, y] which the stretching subunit can stretch in the y-direction.
* `datatype`: The type of manifold, must be set to `"cryo-em"`.

## Running experiments

To run the algorithm on a particular dataset, use

```
python run_experiments.py <path/to/data.pickle> <path/to/configs.json> --generate_plots
```

For example,

```
python run_experiments.py data/rectangle3d_info.pickle configs/configs_rectangle3d.json --generate_plots
```

will run the algorithm on the dataset specified in `data/rectangle3d_info.json` and generate figures for the different experiments. 
To run the experiments without producing figures, simply omit the `--generate_plots` flag.

## Customizing algorithm parameters

The settings for the algorithm used in the experiments from the paper are stored in a `.json` file.

The format of parameters is:

* `sigma`: The width of the kernel for constructing the data graph.
* `n_factors`: The desired number of factors to extract.
* `n_eigenvectors`: The number of eigenvectors to compute.
* `eig_crit`: The threshold for the eigenvalue criterion (see Section 3.1 of the paper for more details).
* `sim_crit`: The threshold for the similarity criterion (see Section 3.1 of the paper for more details).
* `K`: The voting threshold.
* `seed`: A random seed.
* `uniform`: Set to `true` if the data was uniformly sampled, otherwise set to `false`.

## Reproducing figures

To reproduce the figures in the paper, first make sure that the datasets are generated:

```
python generate_data.py params/params_rectangle3d.json params/params_torus.json params/params_cryo-em_x-theta_noisy.json
```

Then run the script `./reproduce_results.sh`. 


Alternatively, you can run the experiments:

```
python factorize.py --data data/rectangle3d_info.pkl --configs configs/configs_rectangle3d.json --outdir results/rectangle3d

python factorize.py --data data/torus_info.pkl --configs configs/configs_torus.json --outdir results/torus

python factorize.py --data data/cryo-em_x-theta_noisy_info.pkl --configs configs/configs_cryo-em_x-theta_noisy.json --outdir results/cryo-em_x-theta_noisy
```

And then run the following:

```
python reproduce_figures.py
```

The figures will be located in the `figures/` folder.
