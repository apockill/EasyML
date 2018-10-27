from setuptools import setup


scripts = ["./easy_ml/freeze_tf_ckpt.py"]

setup(
    name='easy_ml',
    scripts=scripts,
    version='0.1',
    description='This is a helper library with various tools and utils for '
                'training, creating, and preparing models.',
    packages=['easy_ml'],
    install_requires=["tensorflow-gpu"],
    zip_safe=False
)