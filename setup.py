from setuptools import setup

with open('recoengi/requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='recoengi',
    url='https://github.com/ngshya/recoengi',
    author='ngshya',
    author_email='ngshya@gmail.com',
    packages=['recoengi'],
    install_requires=required,
    version='0.0.1',
    license='proprietary',
    description='',
    long_description=open('README.md').read(),
    package_data={
        'notebooks': ['notebooks/*'],
        'static': ['static/*'],
        'sampledata': ['sampledata/*'],
        'scripts': ['scripts/*']
    }
)
