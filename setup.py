from setuptools import setup, find_packages

setup(
    name='LibraryDurKud',
    packages=find_packages(),
    url='https://github.com/hamiddl1/mylib.git',
    description='This is a description for mylib',
    #long_description=open('README.md').read(),
    install_requires=[
        "requests==2.7.0",
        "SomePrivateLib>=0.1.0",
        ],
    dependency_links = [
     "git+git://github.com/hamiddl1/mylib.git#egg=mylib",
    ],
    include_package_data=True,
)
