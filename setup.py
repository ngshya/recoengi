from setuptools import setup

setup(
    name='recoengi',
    url='https://github.com/ngshya/recoengi',
    author='ngshya',
    author_email='ngshya@gmail.com',
    packages=['cf'],
    install_requires=['numpy', 'pandas'],
    version='0.0.1',
    license='proprietary',
    description='',
    long_description=open('README.md').read(),
    package_data={
        'notebooks': ['notebooks/*'],
        'static': ['static/*'],
        'sampledata': ['sampledata/*']
    }
)
