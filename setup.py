from setuptools import setup

setup(
    name='predict_customer_churn',
    author='Damien Michelle',
    author_email='damienmichelle1994@hotmail.com',
    version='1.0',
    packages=['churn'],
    include_package_data=True,
    python_requires='~=3.6',
    description='Predict customer churn',
    license='LICENSE',
    entry_points={
        'console_scripts': ['churn=churn.main:main']
    },
    long_description=open('README.md').read()
)
