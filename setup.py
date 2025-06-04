from setuptools import setup

setup(
        name="breakdance",
        version="0.1",
        packages=["src"],
        install_requires=[
            'opencv-python',
            'setproctitle',
            'loguru',
            'common_ml @ git+ssh://git@github.com/eluv-io/common-ml.git#egg=common_ml',
            'scikit-learn',
            'imagebind @ git+ssh://git@github.com/facebookresearch/ImageBind.git#egg=imagebind',
            'quick_test_py @ git+https://github.com/eluv-io/quick-test-py.git#egg=quick_test_py'
        ]
)