#----------------------------------------------------------
# File simple_bvh_import.py
# Simple bvh exporter
#----------------------------------------------------------
import bpy, os, math, mathutils, time
from mathutils import Vector, Matrix

#
#    class CNode:
#

class CNode:
    def __init__(self, words, parent):
        name = words[1]
        for word in words[2:]:
            name += ' '+word
        
        self.name = name
        self.parent = parent
        self.children = []
        self.head = Vector((0,0,0))
        self.offset = Vector((0,0,0))
        if parent:
            parent.children.append(self)
        self.channels = []
        self.matrix = None
        self.inverse = None
        return

    def __repr__(self):
        return "CNode %s" % (self.name)

    def display(self, pad):
        vec = self.offset
        if vec.length &lt; Epsilon:
            c = '*'
        else:
            c = ' '
        print("%s%s%10s (%8.3f %8.3f %8.3f)" % 
            (c, pad, self.name, vec[0], vec[1], vec[2]))
        for child in self.children:
            child.display(pad+"  ")
        return

    def build(self, amt, orig, parent):
        self.head = orig + self.offset
        if not self.children:
            return self.head
        
        zero = (self.offset.length &lt; Epsilon)
        eb = amt.edit_bones.new(self.name)        
        if parent:
            eb.parent = parent
        eb.head = self.head
        tails = Vector((0,0,0))
        for child in self.children:
            tails += child.build(amt, self.head, eb)
        n = len(self.children)
        eb.tail = tails/n
        self.matrix = eb.matrix.rotation_part()
        self.inverse = self.matrix.copy().invert()
        if zero:
            return eb.tail
        else:        
            return eb.head

#
#    readBvhFile(context, filepath, rot90, scale):
#

Location = 1
Rotation = 2
Hierarchy = 1
Motion = 2
Frames = 3

Deg2Rad = math.pi/180
Epsilon = 1e-5

def readBvhFile(context, filepath, rot90, scale):
    fileName = os.path.realpath(os.path.expanduser(filepath))
    (shortName, ext) = os.path.splitext(fileName)
    if ext.lower() != ".bvh":
        raise NameError("Not a bvh file: " + fileName)
    print( "Loading BVH file "+ fileName )

    time1 = time.clock()
    level = 0
    nErrors = 0
    scn = context.scene
            
    fp = open(fileName, "rU")
    print( "Reading skeleton" )
    lineNo = 0
    for line in fp: 
        words= line.split()
        lineNo += 1
        if len(words) == 0:
            continue
        key = words[0].upper()
        if key == 'HIERARCHY':
            status = Hierarchy
        elif key == 'MOTION':
            if level != 0:
                raise NameError("Tokenizer out of kilter %d" % level)    
            amt = bpy.data.armatures.new("BvhAmt")
            rig = bpy.data.objects.new("BvhRig", amt)
            scn.objects.link(rig)
            scn.objects.active = rig
            bpy.ops.object.mode_set(mode='EDIT')
            root.build(amt, Vector((0,0,0)), None)
            #root.display('')
            bpy.ops.object.mode_set(mode='OBJECT')
            status = Motion
        elif status == Hierarchy:
            if key == 'ROOT':    
                node = CNode(words, None)
                root = node
                nodes = [root]
            elif key == 'JOINT':
                node = CNode(words, node)
                nodes.append(node)
            elif key == 'OFFSET':
                (x,y,z) = (float(words[1]), float(words[2]), float(words[3]))
                if rot90:                    
                    node.offset = scale*Vector((x,-z,y))
                else:
                    node.offset = scale*Vector((x,y,z))
            elif key == 'END':
                node = CNode(words, node)
            elif key == 'CHANNELS':
                oldmode = None
                for word in words[2:]:
                    if rot90:
                        (index, mode, sign) = channelZup(word)
                    else:
                        (index, mode, sign) = channelYup(word)
                    if mode != oldmode:
                        indices = []
                        node.channels.append((mode, indices))
                        oldmode = mode
                    indices.append((index, sign))
            elif key == '{':
                level += 1
            elif key == '}':
                level -= 1
                node = node.parent
            else:
                raise NameError("Did not expect %s" % words[0])
        elif status == Motion:
            if key == 'FRAMES:':
                nFrames = int(words[1])
            elif key == 'FRAME' and words[1].upper() == 'TIME:':
                frameTime = bpy.context.scene.render.fps*float(words[2])
                print(frameTime)
                #frameTime = 1
                status = Frames
                frame = 0
                t = 0
                bpy.ops.object.mode_set(mode='POSE')
                pbones = rig.pose.bones
                for pb in pbones:
                    pb.rotation_mode = 'QUATERNION'
        elif status == Frames:
            addFrame(words, frame, nodes, pbones, scale)
            t += frameTime
            frame += frameTime
           
    fp.close()
    time2 = time.clock()
    print("Bvh file loaded in %.3f s" % (time2-time1))
    return rig

#
#    channelYup(word):
#    channelZup(word):
#

def channelYup(word):
    if word == 'Xrotation':
        return ('X', Rotation, +1)
    elif word == 'Yrotation':
        return ('Y', Rotation, +1)
    elif word == 'Zrotation':
        return ('Z', Rotation, +1)
    elif word == 'Xposition':
        return (0, Location, +1)
    elif word == 'Yposition':
        return (1, Location, +1)
    elif word == 'Zposition':
        return (2, Location, +1)

def channelZup(word):
    if word == 'Xrotation':
        return ('X', Rotation, +1)
    elif word == 'Yrotation':
        return ('Z', Rotation, +1)
    elif word == 'Zrotation':
        return ('Y', Rotation, -1)
    elif word == 'Xposition':
        return (0, Location, +1)
    elif word == 'Yposition':
        return (2, Location, +1)
    elif word == 'Zposition':
        return (1, Location, -1)

#
#    addFrame(words, frame, nodes, pbones, scale):
#

def addFrame(words, frame, nodes, pbones, scale):
    m = 0
    for node in nodes:
        name = node.name
        try:
            pb = pbones[name]
        except:
            pb = None
        if pb:
            for (mode, indices) in node.channels:
                if mode == Location:
                    vec = Vector((0,0,0))
                    for (index, sign) in indices:
                        vec[index] = sign*float(words[m])
                        m += 1
                    pb.location = node.inverse * (scale * vec - node.head)                
                    for n in range(3):
                        pb.keyframe_insert('location', index=n, frame=frame, group=name)
                elif mode == Rotation:
                    mats = []
                    for (axis, sign) in indices:
                        angle = sign*float(words[m])*Deg2Rad
                        mats.append(Matrix.Rotation(angle, 3, axis))
                        m += 1
                    mat = node.inverse * mats[0] * mats[1] * mats[2] * node.matrix
                    pb.rotation_quaternion = mat.to_quat()
                    for n in range(4):
                        pb.keyframe_insert('rotation_quaternion',
                                           index=n, frame=frame, group=name)
    return

#
#    initSceneProperties(scn):
#

def initSceneProperties(scn):
    bpy.types.Scene.MyBvhRot90 = bpy.props.BoolProperty(
        name="Rotate 90 degrees", 
        description="Rotate the armature to make Z point up")
    scn['MyBvhRot90'] = True

    bpy.types.Scene.MyBvhScale = bpy.props.FloatProperty(
        name="Scale", 
        default = 1.0,
        min = 0.01,
        max = 100)
    scn['MyBvhScale'] = 1.0

initSceneProperties(bpy.context.scene)

#
#    class BvhImportPanel(bpy.types.Panel):
#

class BvhImportPanel(bpy.types.Panel):
    bl_label = "BVH import"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        self.layout.prop(context.scene, "MyBvhRot90")
        self.layout.prop(context.scene, "MyBvhScale")
        self.layout.operator("object.LoadBvhButton")

#
#    class OBJECT_OT_LoadBvhButton(bpy.types.Operator):
#

class OBJECT_OT_LoadBvhButton(bpy.types.Operator):
    bl_idname = "OBJECT_OT_LoadBvhButton"
    bl_label = "Load BVH file (.bvh)"

    filepath = bpy.props.StringProperty(name="File Path", 
        maxlen=1024, default="")

    def execute(self, context):
        import bpy, os
        readBvhFile(context, self.properties.filepath, 
            context.scene.MyBvhRot90, context.scene.MyBvhScale)
        return{'FINISHED'}    

    def invoke(self, context, event):
        context.window_manager.add_fileselect(self)
        return {'RUNNING_MODAL'}