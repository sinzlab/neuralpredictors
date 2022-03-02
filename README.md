# Neuralpredictors

[![Test](https://github.com/sinzlab/neuralpredictors/actions/workflows/test.yml/badge.svg)](https://github.com/sinzlab/neuralpredictors/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/sinzlab/neuralpredictors/branch/main/graph/badge.svg)](https://codecov.io/gh/sinzlab/neuralpredictors)
[![Black](https://github.com/sinzlab/neuralpredictors/actions/workflows/black.yml/badge.svg)](https://github.com/sinzlab/neuralpredictors/actions/workflows/black.yml)
[![Mypy](https://github.com/sinzlab/neuralpredictors/actions/workflows/mypy.yml/badge.svg)](https://github.com/sinzlab/neuralpredictors/actions/workflows/mypy.yml)
[![Isort](https://github.com/sinzlab/neuralpredictors/actions/workflows/isort.yml/badge.svg)](https://github.com/sinzlab/neuralpredictors/actions/workflows/isort.yml)
[![PyPI version](https://badge.fury.io/py/neuralpredictors.svg)](https://badge.fury.io/py/neuralpredictors)

[Sinz Lab](https://sinzlab.org/) Neural System Identification Utilities for [PyTorch](https://pytorch.org/).

## How to run the tests :test_tube:

Clone this repository and run the following command from within the cloned repository to run all tests:

```bash
docker-compose run pytest
```

## How to contribute :fire:

Pull requests (and issues) are always welcome. This section describes some
preconditions that pull requests need to fulfill.

### Tests

Please make sure your changes pass the tests. Take a look at the [test running
section](#how-to-run-the-tests-test_tube) for instructions on how to run them. Adding tests
for new code is not mandatory but encouraged.

### Code Style

#### black

This project uses the [black](https://github.com/psf/black) code formatter. You
can check whether your changes comply with its style by running the following
command:

```bash
docker-compose run black
```

Furthermore you can pass a path to the service to have black fix any errors in
the Python modules it finds in the given path.

#### isort

[isort](https://github.com/PyCQA/isort) is used to sort Python imports. You can check the order of imports by running the following command:

```bash
docker-compose run isort
```

The imports can be sorted by passing a path to the service.

### Type Hints

We use [mypy](https://github.com/python/mypy) as a static type checker. Running
the following command will check the code for any type errors:

```bash
docker-compose run mypy
```

It is not necessary (but encouraged) to add type hints to new code but please
make sure your changes do not produce any mypy errors.

Note that only modules specified in the `mypy-files.txt` file are checked by
mypy. This is done to be able to add type hints gradually without drowning in
errors. If you want to add type annotations to a previously unchecked module
you have to add its path to `mypy-files.txt`.
