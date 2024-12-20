from setuptools import Extension, setup

module = Extension("geo_capi", sources=['symnmfmodule.c'])
setup(name='symnmf_module',
     version='1.0',
     description='Wrapper for Symnmf algorithm in C',
     ext_modules=[module])