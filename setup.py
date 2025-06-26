from setuptools import setup, find_packages
import os

# Function to read the version from __version__.py
def get_version(package_name):
    version_file = os.path.join(os.path.dirname(__file__), 'src', package_name, '__version__.py')
    with open(version_file) as f:
        exec(f.read())
    return locals()['__version__']

setup(name='stuf',
    version=get_version('stuf'),  # Dynamically read the version
    description='Simple Tools and Useful Functions (STUF) that make easy some common tasks in the analysis of single cell and spatial transcriptomics data.',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',  
    author='Patrick Cahan',
    author_email='patrick.cahan@gmail.com',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'matplotlib',
        'scanpy',        
    ],
    project_urls={
        'Documentation': 'https://stuf.readthedocs.io/en/latest/',
        'Source': 'https://github.com/CahanLab/stuf'
    },
)