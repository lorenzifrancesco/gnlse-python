from setuptools import find_packages
from setuptools import setup

def parse_requirements(path):
    reqs = []
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('-r '):
                nested = line.split(maxsplit=1)[1]
                reqs.extend(parse_requirements(nested))
                continue
            reqs.append(line)
    return reqs


reqs = parse_requirements('requirements.txt')

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gnlse',
    version='2.0.0',
    url='https://github.com/WUST-FOG/gnlse-python',
    author='Redman, P., Zatorska, M., Pawlowski, A., Szulc, D., '
           'Majchrowska, S., Tarnowski, K.',
    description='gnlse-python is a Python set of scripts for solving '
                'Generalized Nonlinear Schrodringer Equation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=reqs,
)
