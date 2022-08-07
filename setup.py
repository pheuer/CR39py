import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='CR39py',
    version='0.0.1',
    author='Peter Heuer',
    author_email='pheu@lle.rochester.edu',
    description='Code for analyzing CR39 particle track data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pheuer/CR39py',
    project_urls = {
        "Bug Tracker": "https://github.com/pheuer/CR39py/issues"
    },
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'h5py', 'matplotlib','fast_histogram', 'astropy'],
)
