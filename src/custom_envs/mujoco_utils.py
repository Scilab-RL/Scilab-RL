import numpy as np
import mujoco
from mujoco import MjData, MjModel


def get_geom_xpos(model, data, name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return data.geom_xpos[geom_id]


def get_geom_xvelp(model, data, name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    jacp = get_geom_jacp(model, data, geom_id)
    xvelp = jacp @ data.qvel
    return xvelp


def get_geom_xvelr(model, data, name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    jacp = get_geom_jacr(model, data, geom_id)
    xvelp = jacp @ data.qvel
    return xvelp


def get_geom_jacp(model, data, geom_id):
    """Return the Jacobian' translational component of the end-effector of
    the corresponding geom id.
    """
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacGeom(model, data, jacp, None, geom_id)

    return jacp


def get_geom_jacr(model, data, geom_id):
    """Return the Jacobian' rotational component of the end-effector of
    the corresponding geom id.
    """
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacGeom(model, data, None, jacr, geom_id)

    return jacr


def get_geom_xmat(model: MjModel, data: MjData, name: str):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return data.geom_xmat[geom_id].reshape(3, 3)
