#!/usr/bin/python3
# -*- coding: utf8 -*-

from distutils.core import setup
setup(
  name = 'neural_wrappers',         # How you named your package folder (MyLib)
  packages = ['neural_wrappers'],   # Chose the same as "name"
  version = '0.2',      # Start with a small number and increase it with every change you make
  license='wtfpl',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Generic neural networks high level wrapper for PyTorch',   # Give a short description about your library
  author = 'Mihai Cristian Pîrvu',                   # Type in your name
  author_email = 'mihaicristianpirvu@gmail.com',      # Type in your E-Mail
  url = 'https://gitlab.com/mihaicristianpirvu/neural-wrappers',   # Provide either the link to your github or to your website
  download_url = 'https://gitlab.com/mihaicristianpirvu/neural-wrappers/-/archive/v0.2/neural-wrappers-v0.2.zip',    # I explain this later on
  keywords = ['PyTorch', 'neural network', 'high level api'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)