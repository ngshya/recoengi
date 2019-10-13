from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='recoengi',
    url='https://github.com/ngshya/recoengi',
    author='ngshya',
    author_email='ngshya@gmail.com',
    packages=['recoengi', 'recoengi.cf', 'recoengi.cv', 'recoengi.sampledata'],
    install_requires=required,
    version='0.0.10',
    license='proprietary',
    description='',
    long_description=open('README.md').read(),
    package_data={
        'recoengi': ['*'],
        'recoengi': ['sampledata/*'],
        'notebooks': ['*'],
        'static': ['*'],
        'scripts': ['*']
    }
)
