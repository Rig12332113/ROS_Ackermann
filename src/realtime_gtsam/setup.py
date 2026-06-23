from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'realtime_gtsam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rick',
    maintainer_email='rig12332113@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gtsam = realtime_gtsam.realtime_gtsam:main',
            'gtsam_pc = realtime_gtsam.realtime_gtsam_with_pointcloud:main',
            'gtsam_octo = realtime_gtsam.realtime_gtsam_with_octomap:main',
        ],
    },
)
