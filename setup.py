import os
from setuptools import setup, find_packages

def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname)) as fh:
            return fh.read()
    except IOError:
        return ''

requirements = read('requirements.txt').splitlines()


setup(name='deep_audio_features',
      version='0.1.9',
      description='Extract supervised deep features using CNN audio classifiers',
      url='https://github.com/tyiannak/deep_audio_features',
      author='Theodoros Giannakopoulos',
      author_email='tyiannak@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=requirements,
      )
