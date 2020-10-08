from setuptools import setup

setup(
    name='GRImpulsiveWaves',
    version='0.2',
    packages=['grimpulsivewaves', 'grimpulsivewaves.waves', 'grimpulsivewaves.plotting', 'grimpulsivewaves.coordinates',
              'grimpulsivewaves.integrators'],
    package_dir={'': 'src'},
    url='',
    license='',
    author='Daniel Rod',
    author_email='daniel.rod@seznam.cz',
    description='Visualisation of geodesics in impulsive spacetimes using refraction equations.'
)
