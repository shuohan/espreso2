from setuptools import setup

version = '0.1.0'

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
      python_requires='>=3.8.3',
      scripts=['scripts/train.py'],
      long_description=long_desc,
      long_description_content_type='text/markdown',
      url='https://github.com/shuohan/espreso2.git',
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent']
      )
