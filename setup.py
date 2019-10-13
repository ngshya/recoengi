from setuptools import setup

with open('./recoengi/requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='recoengi',
    url='https://github.com/ngshya/recoengi',
    author='ngshya',
    author_email='ngshya@gmail.com',
    packages=['recoengi', 'recoengi.cf', 'recoengi.cv', 'sampledata'],
    install_requires=required,
    version='0.0.7',
    license='proprietary',
    description='',
    long_description=open('README.md').read(),
    package_data={
        'recoengi': ['*'],
        'notebooks': ['*'],
        'static': ['*'],
        'sampledata': ['*'],
        'scripts': ['*']
    }, 
    data_files=[('recoengi', ['recoengi/requirements.txt']),
                ('sampledata', ['sampledata/movie_ratings_train.pickle', 
                                'sampledata/movie_ratings_test.pickle'])]
)
