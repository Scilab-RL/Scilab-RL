"""adapted from https://github.com/tensorflow/models/tree/master/research/efficient-hrl/environments/maze_env.py
and https://github.com/tensorflow/models/tree/master/research/efficient-hrl/environments/maze_env_utils.py"""
import os
import tempfile
import xml.etree.ElementTree as ET
from shutil import copyfile


def create_xml(xml_path, maze_id=None, height=0.5, size_scaling=8):
    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    structure = construct_maze(maze_id)  # Get the maze structure that corresponds to the maze_id
    elevated = any(-1 in row for row in structure)  # Elevate the maze to allow for falling.
    blocks = any(any(can_move(r) for r in row)for row in structure)  # Are there any movable blocks?

    # determine the robots position
    for i, _ in enumerate(structure):
        for j, _ in enumerate(structure[0]):
            if structure[i][j] == 'r':
                torso_x = j * size_scaling
                torso_y = i * size_scaling

    height_offset = 0.
    if elevated:
        # Increase initial z-pos of ant.
        height_offset = height * size_scaling
        torso = tree.find(".//body[@name='torso']")
        torso.set('pos', '0 0 %.2f' % (0.75 + height_offset))
    if blocks:
        # If there are movable blocks, change simulation settings to perform
        # better contact detection.
        default = tree.find(".//default")
        default.find('.//geom').set('solimp', '.995 .995 .01')

    for i, _ in enumerate(structure):
        for j, _ in enumerate(structure[0]):
            if elevated and structure[i][j] not in [-1]:
                # Create elevated platform.
                ET.SubElement(
                    worldbody, "geom",
                    name="elevated_%d_%d" % (i, j),
                    pos="%f %f %f" % (j * size_scaling - torso_x,
                                      i * size_scaling - torso_y,
                                      height / 2 * size_scaling),
                    size="%f %f %f" % (0.5 * size_scaling,
                                       0.5 * size_scaling,
                                       height / 2 * size_scaling),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.9 0.9 0.9 1",
                )
            if structure[i][j] == 1:  # Unmovable block.
                # Offset all coordinates so that robot starts at the origin.
                ET.SubElement(
                    worldbody, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (j * size_scaling - torso_x,
                                      i * size_scaling - torso_y,
                                      height_offset +
                                      height / 2 * size_scaling),
                    size="%f %f %f" % (0.5 * size_scaling,
                                       0.5 * size_scaling,
                                       height / 2 * size_scaling),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.4 0.4 0.4 1",
                )
            elif can_move(structure[i][j]):  # Movable block.
                # The "falling" blocks are shrunk slightly and increased in mass to
                # ensure that it can fall easily through a gap in the platform blocks.
                falling = can_move_z(structure[i][j])
                shrink = 0.99 if falling else 1.0
                moveable_body = ET.SubElement(
                    worldbody, "body",
                    name="moveable_%d_%d" % (i, j),
                    pos="%f %f %f" % (j * size_scaling - torso_x,
                                      i * size_scaling - torso_y,
                                      height_offset +
                                      height / 2 * size_scaling),
                )
                ET.SubElement(
                    moveable_body, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="0 0 0",
                    size="%f %f %f" % (0.5 * size_scaling * shrink,
                                       0.5 * size_scaling * shrink,
                                       height / 2 * size_scaling),
                    type="box",
                    material="",
                    mass="0.001" if falling else "0.0002",
                    contype="1",
                    conaffinity="1",
                    rgba="0.9 0.1 0.1 1"
                )
                if can_move_x(structure[i][j]):
                    ET.SubElement(
                        moveable_body, "joint",
                        armature="0",
                        axis="1 0 0",
                        damping="0.0",
                        limited="true" if falling else "false",
                        range="%f %f" % (-size_scaling, size_scaling),
                        margin="0.01",
                        name="moveable_x_%d_%d" % (i, j),
                        pos="0 0 0",
                        type="slide"
                    )
                if can_move_y(structure[i][j]):
                    ET.SubElement(
                        moveable_body, "joint",
                        armature="0",
                        axis="0 1 0",
                        damping="0.0",
                        limited="true" if falling else "false",
                        range="%f %f" % (-size_scaling, size_scaling),
                        margin="0.01",
                        name="moveable_y_%d_%d" % (i, j),
                        pos="0 0 0",
                        type="slide"
                    )
                if can_move_z(structure[i][j]):
                    ET.SubElement(
                        moveable_body, "joint",
                        armature="0",
                        axis="0 0 1",
                        damping="0.0",
                        limited="true",
                        range="%f 0" % (-height_offset),
                        margin="0.01",
                        name="moveable_z_%d_%d" % (i, j),
                        pos="0 0 0",
                        type="slide"
                    )

    torso = tree.find(".//body[@name='torso']")
    geoms = torso.findall(".//geom")
    for geom in geoms:
        if 'name' not in geom.attrib:
            raise Exception("Every geom of the torso must have a name "
                            "defined")

    # copy subgoal_viz.xml to the /tmp folder so it is available for the created environment xml
    copyfile(os.path.join(os.path.dirname(xml_path), 'subgoal_viz.xml'), '/tmp/subgoal_viz.xml')

    _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
    tree.write(file_path)

    return file_path


class Move:
    X = 11
    Y = 12
    Z = 13
    XY = 14
    XZ = 15
    YZ = 16
    XYZ = 17


def can_move_x(movable):
    return movable in [Move.X, Move.XY, Move.XZ, Move.XYZ]


def can_move_y(movable):
    return movable in [Move.Y, Move.XY, Move.YZ, Move.XYZ]


def can_move_z(movable):
    return movable in [Move.Z, Move.XZ, Move.YZ, Move.XYZ]


def can_move(movable):
    return can_move_x(movable) or can_move_y(movable) or can_move_z(movable)


def construct_maze(maze_id='Maze'):
    if maze_id == 'Maze':
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == 'Push':
        structure = [
            [1, 1, 1, 1, 1],
            [1, 0, 'r', 1, 1],
            [1, 0, Move.XY, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == 'Fall':
        structure = [
            [1, 1, 1, 1],
            [1, 'r', 0, 1],
            [1, 0, Move.YZ, 1],
            [1, -1, -1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
    else:
        raise NotImplementedError(
            'The provided MazeId %s is not recognized' % maze_id)

    return structure
