from setuptools import find_packages, setup

package_name = 'open_cyber_glove_retarget_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zimo',
    maintainer_email='jessey.li@cyberorigin.ai',
    description='ROS2 package for OpenCyberGlove retargeting',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'retarget_node = open_cyber_glove_retarget_ros2.retarget_node:main',
        ],
    },
)
