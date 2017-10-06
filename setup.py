from setuptools import setup

setup(name='gym_air_hockey',
      version='0.0.1',
      description='OpenAI Gym Environment Wrapper for Air Hockey Game Simulator',
      url='https://github.com/arakhmat/gym-air-hockey',
      author='Akhmed Rakhmati',
      author_email='akhmed.rakhmati@gmail.com',
      install_requires=['gym',
                        'pygame',
                        'numpy',
                        'opencv-python']
)