from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name="ocd",
    version="0.1",
    description="Tool to calculate immunoglobulin inter-domain orientations",
    author="Valentin J. Hoerschinger",
    author_email="valentin.hoerschinger@uibk.ac.at",
    include_package_data=True,
    zip_safe=False,
    licence = "GPLv3"
    long_description=readme(),
    url = "https://github.com/liedllab/OCD"
    packages=['ocd'],
    py_modules=["ocd.visualize", "ocd.calculation","ocd.OCD"],
    install_requires=[
          'pandas',
          'mdtraj',
          'pytraj',
          'numpy',
          'matplotlib',
    ],
    entry_points={
    'console_scripts': [
        'OCD=ocd.OCD:main',
    ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)