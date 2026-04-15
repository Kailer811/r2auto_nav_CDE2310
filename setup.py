from setuptools import setup

package_name = 'auto_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_dir={package_name: '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nus',
    maintainer_email='nus@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'r2mover = auto_nav.r2mover:main',
            'r2moverotate = auto_nav.r2moverotate:main',
            'r2scanner = auto_nav.r2scanner:main',
            'r2occupancy = auto_nav.r2occupancy:main',
            'r2occupancy2 = auto_nav.r2occupancy2:main',
            'r2auto_nav = auto_nav.r2auto_nav:main',
	    'auto_nav2 = auto_nav.auto_nav2:main',
	    'auto_nav_draft = auto_nav.auto_nav_draft:main',
            'aruco_node = auto_nav.aruco_node:main',
            'ros2_nav_aruco = auto_nav.ros2_nav_aruco:main',
            'nav2_go_to_pose = auto_nav.nav2_go_to_pose:main',
            'RamIsBetter = auto_nav.RamIsBetter:main',
            'aruco_precision_dock = auto_nav.aruco_precision_dock:main',
            'aruco_pid_dock = auto_nav.aruco_pid_dock:main',
            'aruco_nav_fsm = auto_nav.aruco_nav_fsm:main',
            'ros2_nav = auto_nav.ros2_nav:main',
            'ros2_nav_kms = auto_nav.ros2_nav_kms:main'
        ],
    },
)
