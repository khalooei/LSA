from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='LayersSustainabilityAnalysis',
    version='1.0.1',
    url='https://github.com/khalooei/LSA',
    license='MIT',
    description='A Python library that analyzes the layer sustainability of neural networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Mohammad Khalooei',
    packages=['LayerSustainabilityAnalysis'],
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
    ],
    project_urls={
        'Changelog': ('https://github.com/khalooei/LSA/README.md'),
        'Docs': 'https://github.com/khalooei/LSA/README.md',
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],

)