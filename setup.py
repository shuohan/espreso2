from setuptools import setup

version = '0.3.2'

with open('README.md') as readme:
    long_desc = readme.read()

setup(name='espreso2',
      description='An algorithm to estimate an MR slice profiles by learning '
                  'to match internal patch distributions',
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      version=version,
      packages=['espreso2'],
      license='GPLv3',
      python_requires='>=3.7.10',
      entry_points={
          'console_scripts': [
              'espreso2-train=espreso2.exec:train',
              'espreso2-fwhm=espreso2.exec:fwhm',
          ]
      },
      long_description=long_desc,
      install_requires=[
          'torch>=1.8.1',
          'numpy',
          'scipy',
          'nibabel',
          'matplotlib',
          'ptxl@git+https://gitlab.com/shan-deep-networks/ptxl@0.3.2',
          'sssrlib@git+https://github.com/shuohan/sssrlib@0.3.0'
      ],
      long_description_content_type='text/markdown',
      url='https://github.com/shuohan/espreso2',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: OS Independent'
      ]
      )
