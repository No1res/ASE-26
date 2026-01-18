# raacs/core/roles.py
from enum import Enum

class Role(Enum):
    """
    代码角色枚举 - 9 个角色对应 9 种 Morphing 策略。
    """
    TEST = "TEST"
    NAMESPACE = "NAMESPACE"
    INTERFACE = "INTERFACE"
    SCHEMA = "SCHEMA"
    ADAPTER = "ADAPTER"
    CONFIG = "CONFIG"
    SCRIPT = "SCRIPT"
    UTIL = "UTIL"
    LOGIC = "LOGIC"
    UNKNOWN = "UNKNOWN"

class RoleSource(Enum):
    """
    角色来源 - 用于追踪角色判定的信号来源。
    """
    INITIAL_FRAMEWORK = "initial_framework"
    INITIAL_DECORATOR = "initial_decorator"
    INITIAL_NAME = "initial_name"
    INHERITED = "inherited"
    PROPAGATED = "propagated"
    STRUCTURAL = "structural"
    UNKNOWN = "unknown"

# 角色来源强度映射 - 用于判断覆盖优先级。
ROLE_SOURCE_STRENGTH = {
    RoleSource.INITIAL_FRAMEWORK: 0.95,
    RoleSource.INITIAL_DECORATOR: 0.90,
    RoleSource.INHERITED: 0.85,
    RoleSource.PROPAGATED: 0.75,
    RoleSource.STRUCTURAL: 0.65,
    RoleSource.INITIAL_NAME: 0.50,
    RoleSource.UNKNOWN: 0.0,
}

# 角色兼容性矩阵（九个角色两两之间的兼容度，省略不常用的组合）。
ROLE_COMPATIBILITY = {
    (Role.SCHEMA, Role.CONFIG): 0.9,
    (Role.SCHEMA, Role.INTERFACE): 0.8,
    (Role.CONFIG, Role.INTERFACE): 0.7,
    (Role.UTIL, Role.LOGIC): 0.5,
    (Role.ADAPTER, Role.LOGIC): 0.4,
    (Role.ADAPTER, Role.UTIL): 0.5,
    (Role.SCHEMA, Role.LOGIC): 0.2,
    (Role.SCHEMA, Role.UTIL): 0.3,
    (Role.SCHEMA, Role.ADAPTER): 0.2,
    (Role.CONFIG, Role.LOGIC): 0.3,
    (Role.INTERFACE, Role.LOGIC): 0.4,
    (Role.TEST, Role.LOGIC): 0.1,
    (Role.SCRIPT, Role.LOGIC): 0.3,
    (Role.NAMESPACE, Role.SCHEMA): 0.5,
}

class SignalWeight:
    """
    不同信号的权重定义（用于角色判定）。
    """
    FRAMEWORK = 4.0
    STRUCTURE = 2.5
    PATH_HINT = 1.5
    NAME_HINT = 1.0
    INHERITANCE = 3.5

__all__ = [
    "Role",
    "RoleSource",
    "ROLE_SOURCE_STRENGTH",
    "ROLE_COMPATIBILITY",
    "SignalWeight",
]