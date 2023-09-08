#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from functools import wraps
from typing import Callable, Type

from ..config import __all__ as all_API_methods
from .base import API

# Defines a class decorator to make wraps any API methods so that the Master Address
# and the Master Port are set in order to allow the process group to initialize
# correctly.

env_variables_set: bool = False


def set_master_addr_port_env_variables(func):
    @wraps(func)
    def wrapper(self):
        global env_variables_set
        if not env_variables_set:
            env_variables_set = True  # must be done before actually setting the variable to prevent stackoverflow
            os.environ["MASTER_ADDR"] = self.master_address()
            os.environ["MASTER_PORT"] = str(self.port())
        return func(self)

    return wrapper


def decorate_methods(cls: Type[API], func_to_apply: Callable) -> Type[API]:
    for obj_name in dir(cls):
        if obj_name in all_API_methods:
            decorated = func_to_apply(getattr(cls, obj_name))
            setattr(cls, obj_name, decorated)

    return cls


def AutoMasterAddressPort(cls: Type[API]) -> Type[API]:
    return decorate_methods(cls, func_to_apply=set_master_addr_port_env_variables)