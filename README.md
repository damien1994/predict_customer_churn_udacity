Predict Customer Churn with Udacity
======================

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
[WARNING : copy paste from project introduction of udacity]

In this project, you will implement your learnings to identify credit card customers that are most likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

This project will give you practice using your skills for testing, logging, and best coding practices from this lesson. It will also introduce you to a problem data scientists across companies face all the time. How do we identify (and later intervene with) customers who are likely to churn?

You can find data input [here](churn/data/bank_data.csv)

## Running Files 
How do you run your files? What should happen when you run your files?

### Local Developement (run, testing, linting)
Python >= 3.8

```bash
virtualenv -p python3.8 venv
source venv/bin/activate
pip3 install -r requirements.txt

# Local run 
make run
or 
churn --input_fle <your csv file>

# Local test
make tests

# Local pylint
make linter
```

All logs are contained in .log files:

- [churn_library.log](churn/logs/churn_library.log) : logs for run execution
- [test_churn_library.log](churn/tests/logs/test_churn_library.log) : logs for tests exection


### Docker (feature on creation)
not developed yet

## REFERENCES
- [Udacity project instructions and lessons](https://classroom.udacity.com/nanodegrees/nd0821/parts/1f633c06-5f45-4309-a1cb-ca11112105f5/modules/240dd233-3306-4d38-9de0-3fc275f13913/lessons/11d78371-6574-414d-b7e9-3b28fb008f1c/concepts/223da2b2-c4fa-42a8-9104-942fd378c780)
- [Examples on github of pytest use for data preparation](https://github.com/eugeneyan/testing-ml/blob/master/tests/data_prep/test_categorical.py).
These examples will inspire the way to create our unit test for data prepration
- [Examples on github of pytest use for ML model testing](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/metrics/tests/test_score_objects.py).
These examples will inspire the way to create our unit test for ML

## TO DO
- unit tests with pytest
- logging info vs logging error ? 
- deal with option between retrain model or load pkl
- docker : optional (personal ambition)
- github actions : optional (personal ambition)


