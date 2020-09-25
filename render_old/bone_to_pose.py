import bpy
from bpy import context

remove_constraints = True
scene = context.scene
rig1 = scene.objects["0000015"]

rig2 = scene.objects["Cube.001"]
if not rig2.animation_data:
    rig2.animation_data_create()
rig2.animation_data.action = None

# add copy transform constraint to each bone
for pb in rig2.pose.bones:
    ct = pb.constraints.get(pb.name)    
    if ct is not None:
        ct.influence = 1
        continue
    ct = pb.constraints.new('COPY_TRANSFORMS')
    ct.name = pb.name
    ct.target = rig1
    ct.subtarget = pb.name

action = rig1.animation_data.action

f = action.frame_range.x
# add a keyframe to each frame of new rig
while f < action.frame_range.y:
    scene.frame_set(f)
    for pb in rig2.pose.bones:
        #pb2 = rig1.pose.bones.get(pb.name)
        m = rig2.convert_space(pb, pb.matrix, to_space='LOCAL')
        if pb.rotation_mode == 'QUATERNION':
            pb.rotation_quaternion = m.to_quaternion()
            pb.keyframe_insert("rotation_quaternion", frame=f)
        else:

        # add rot mode checking 
            pb.rotation_euler = m.to_euler(pb.rotation_mode)
            pb.keyframe_insert("rotation_euler", frame=f)
        pb.location = m.to_translation()

        pb.keyframe_insert("location", frame=f)
    f += 1

# set constraints to zero or remove entirely.
for pb in rig2.pose.bones:
    ct = pb.constraints.get(pb.name)    
    if ct is not None:
        if remove_constraints:
            pb.constraints.remove(ct)
        else:
            ct.influence = 0