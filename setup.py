from setuptools import setup

setup(
    name='GRImpulsiveWaves',
    version='0.3.2',
    packages=['grimpulsivewaves', 'grimpulsivewaves.waves', 'grimpulsivewaves.plotting', 'grimpulsivewaves.coordinates',
              'grimpulsivewaves.integrators'],
    package_dir={'': 'src'},
    url='',
    license='GNU GPLv3',
    author='Daniel Rod',
    author_email='daniel.rod@seznam.cz',
    description='Visualisation of geodesics in impulsive spacetimes using refraction equations.',
    long_description='GRImpulsiveWaves is aimed to visualise geodesic motion in exact solutions of impulsive wave spacetimes, using refraction equations in Cut and Paste ',
    python_requires='>=3.6',
    install_requires=[
          'numpy', 'scipy', 'matplotlib'
      ],
)
