from setuptools import setup, find_packages
setup(
    name = 'entropy_estimators',
    version = '0.0.1',
    description = 'Estimators for entropy and other information theoretic quantities of continuous distributions',
    author = 'Paul Brodersen',
    author_email = 'paulbrodersen+entropy_estimators@gmail.com',
    url = 'https://github.com/paulbrodersen/entropy_estimators',
    download_url = 'https://github.com/paulbrodersen/entropy_estimators/archive/0.0.1.tar.gz',
    keywords = ['entropy', 'Shannon information', 'mutual information', 'transfer entropy'],
    classifiers = [ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    platforms=['Platform Independent'],
    packages=find_packages(),
    install_requires=['scipy', 'numpy'],
)
