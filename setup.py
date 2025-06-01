from setuptools import setup, find_packages

setup(
    name='ris-secrecy-optimization',
    version='1.0.0',
    author='Robert Kuku Fotock',
    author_email='fotockrobert@gmail.com',
    description='A project for RIS Secrecy Optimization in wireless networks.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bigdoll/ris-secrecy-optimization',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'cvxpy',
        'mosek',
    ],
    entry_points={
        'console_scripts': [
            'ris-secrecy-optimization=main:main',
        ],
    },
     project_urls={
        'Documentation': 'https://github.com/bigdoll/ris-secrecy-optimization/README.md',
        'Source': 'https://github.com/bigdoll/ris-secrecy-optimization',
        'Tracker': 'https://github.com/bigdoll/ris-secrecy-optimization/issues',
        'Google Scholar': 'https://scholar.google.com/citations?user=2ADrhokAAAAJ&hl=en&oi=ao',
        'Orcid': 'https://orcid.org/0009-0005-0232-3596',
    },
)
