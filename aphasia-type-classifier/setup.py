from setuptools import setup, find_packages

setup(
    name='aphasia-type-classifier',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A classifier for identifying types of aphasia in Greek-speaking patients based on linguistic features.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/aphasia-type-classifier',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'nltk',
        'streamlit',
        'matplotlib',
        'seaborn',
        'joblib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)