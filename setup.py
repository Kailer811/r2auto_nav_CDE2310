from setuptools import setup


package_name = 'auto_nav'


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_dir={package_name: 'Software'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
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
            'aruco_nav_fsm = auto_nav.aruco_nav_fsm:main',
            'nav2_dock = auto_nav.Nav2_dock:main',
            'nav2_go_to_pose = auto_nav.nav2_go_to_pose:main',
            'r2auto_nav = auto_nav.r2auto_nav:main',
            'r2mover = auto_nav.r2mover:main',
            'r2moverotate = auto_nav.r2moverotate:main',
            'r2occupancy = auto_nav.r2occupancy:main',
            'r2occupancy2 = auto_nav.r2occupancy2:main',
            'r2scanner = auto_nav.r2scanner:main',
            'ros2_aruco = auto_nav.ros2_aruco:main',
            'ros2_camera = auto_nav.ros2_camera:main',
            'ros2_nav = auto_nav.ros2_nav:main',
            'servo_motor_run = auto_nav.servo_motor_run:main',
        ],
    },
)
