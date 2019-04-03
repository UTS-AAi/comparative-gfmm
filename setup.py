from setuptools import setup, find_packages

setup(
    name='comparative-gfmm',
    version='1.0.1',
    description='Implementation of classifiers based on hyper-box representation written as a Python library',
    author='Thanh Tung Khuat and Bogdan Gabrys',
    author_email=['thanhtung.khuat@student.uts.edu.au', 'bogdan.gabrys@uts.edu.au'],
    url='https://github.com/thanhtung09t2/Hyperbox-classifier',
    download_url='https://github.com/thanhtung09t2/Hyperbox-classifier.git',
    packages=find_packages(),
    include_package_data = True,
    install_requires = ['setuptools'],
    classifiers=[
        'Environment :: Desktop Environment',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python'
    ]
)

