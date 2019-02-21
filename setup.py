from setuptools import setup

setup(name='graph_minimap',
      version='0.0.1',
      description='Rough graph minimap',
      url='http://github.com/uio-bmi/graph_minimap',
      author='Ivar Grytten and Knut Rand',
      author_email='',
      license='MIT',
      zip_safe=False,
      install_requires=['numpy', 'python-coveralls', 'numba', 'pathos', 'sortedcontainers',
                        'pyfaidx', 'scikit-bio', 'tqdm'],
      classifiers=[
            'Programming Language :: Python :: 3'
      ],
      #entry_points = {
      #'console_scripts': ['two_step_graph_mapper=two_step_graph_mapper.command_line_interface:main'],
      #}
)
