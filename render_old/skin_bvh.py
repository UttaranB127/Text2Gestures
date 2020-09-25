import bpy 

# HELPER FUNCTION THAT CREATE EDGE BASED ON SUPPLIED POSITIONS
def createEdge(coord1 = (-1.0, 1.0, 0.0), coord2 = (-1.0, -1.0, 0.0)):
    
    Verts = [coord1, coord2]
    Edges = [[0,1]]
    
    profile_mesh = bpy.data.meshes.new("Edge_Profile_Data")
    profile_mesh.from_pydata(Verts, Edges, [])
    profile_mesh.update()
    
    profile_object = bpy.data.objects.new("Edge_Profile", profile_mesh)
    profile_object.data = profile_mesh
    
    scene = bpy.context.scene
    scene.objects.link(profile_object)
    profile_object.select = True

# ACTUAL FUNCTION THAT TURN BONES INTO EDGES
def boneToEdges(armature_name):
    
    myRig = bpy.data.objects[armature_name]
    
    # which armature to work on
    #myRig = bpy.data.objects['Armature']
    #myRig = bpy.data.objects['metarig']
    
    # this actually return STRING NAME of bone
    boneNames = myRig.data.bones.keys()
    
    # the actual data to each 
    myBones = myRig.data.bones
    
    for i, bone in enumerate(myBones):
        #print(bone.name, bone.vector)
        # every bone has HEAD and TAIL
    
        #loc = bone.vector
        head_loc = bone.head_local
        tail_loc = bone.tail_local
        
        createEdge(coord1 = head_loc, coord2 = tail_loc)
        
boneToEdges("000015")

# Get all objects named "NAME*" and put it in a list
mesh = bpy.data.objects
sel = [item for item in mesh if "Edge_Profile" in item.name]

for item in sel:
    item.select = True

bpy.ops.object.join()