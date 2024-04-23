# -*- coding: utf-8 -*-

import os

from distutils.command.build import build
from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'README.md')) as f:
    README = f.read()

REQUIREMENTS = [
    'numpy',
    'astropy',
    'pymoc',
    'healpy',
    'sfdmap2',
    'seaborn',
    'matplotlib-venn',
    'pyyaml',
    'pyregion==1.2.0',
    'humanfriendly'
]


# class CustomBuild(build):
#     """Build class to build the database"""
#     def run(self):
#         # Build the databse
#         import database_builder
#         database_builder.build_base()
#         # Process with the standard build
#         build.run(self)

setup(
    name="herschelhelp_internal",
    version="1.0.4",
    description="HELP project internal code",
    long_description=README,
    author="Yannick Roehlly",
    author_email="yannick.roehlly@lam.fr",
    license='MIT',
    install_requires=REQUIREMENTS,
    package_data={
        'herschelhelp_internal/sfd_data': ['*.fits'],
    },
    packages=find_packages(exclude=['database_builder']),
    # cmdclass={'build': CustomBuild},
)
