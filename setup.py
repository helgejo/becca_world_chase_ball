from setuptools import setup

setup(
    name='becca_world_chase_ball',
    version='0.8.0',
    description='A world where a dog tries to catch a ball',
    url='http://github.com/brohrer/becca_world_chase_ball',
    download_url='https://github.com/brohrer/becca/archive/master.zip',
    author='Brandon Rohrer',
    author_email='brohrer@gmail.com',
    license='MIT',
    packages=['becca_world_chase_ball'],
    include_package_data=True,
    install_requires=['becca', 'becca_test'],
    zip_safe=False)
