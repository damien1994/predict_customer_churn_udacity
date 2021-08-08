Predict Customer Churn with Udacity
======================

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Your project description here.


## Running Files 
How do you run your files? What should happen when you run your files?

### Local Developement 
Python >= 3.8

```bash
virtualenv -p python3.8 venv
source venv/bin/activate
pip3 install -r requirements.txt

make run
or 
churn --input_fle <your csv file>
```

### Docker (feature on creation)
not developed yet

## REFERENCES
- [Examples on github of pytest use for data preparation](https://github.com/eugeneyan/testing-ml/blob/master/tests/data_prep/test_categorical.py).
These examples will inspire the way to create our unit test for data prepration
- [Examples on github of pytest use for ML model testing](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/metrics/tests/test_score_objects.py).
These examples will inspire the way to create our unit test for ML

## TO DO
- refacto utils into python classes : done
- try / except blocks : done
- unit tests with pytest
- lint : add to makefile
- makefile : add linter and test to makefile
- logging info vs logging error ? 
- deal with option between retrain model or load pkl
- docker : optional (personal ambition)
- github actions : optional (personal ambition)


