#!BPY

import bpy
import bpy_extras
import random
import os
from tkinter import filedialog

PARENT_DIR = filedialog.askdirectory(title='Choose Output Folder',initialdir='./')

NUM_IMAGES = 20

with open(PARENT_DIR+'/Lens_Dataset/data.csv', "w") as myfile:
            csv_line = 'ID,cornerTL_x,cornerTL_y,cornerTR_x,cornerTR_y,cornerBL_x,cornerBL_y,cornerBR_x,cornerBR_y\n'
            myfile.write(csv_line)

# Load the Blender scene
bpy.ops.wm.open_mainfile(filepath=PARENT_DIR+'/Lens_Dataset_Scene.blend')

# Enable the compositor
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
nodes = tree.nodes
links = tree.links

# Remove existing nodes (optional, to clear existing setup)
for node in nodes:
    nodes.remove(node)

# Create Render Layers node (input)
render_layers = nodes.new(type="CompositorNodeRLayers")
render_layers.location = (-300, 0)

# Create Lens Distortion node
lens_distortion = nodes.new(type="CompositorNodeLensdist")
lens_distortion.location = (0, 0)
lens_distortion.inputs["Distortion"].default_value = 0.0  # Adjust for more or less distortion
lens_distortion.inputs["Dispersion"].default_value = 0.00  # Chromatic aberration effect

# Create Composite node (output)
composite = nodes.new(type="CompositorNodeComposite")
composite.location = (300, 0)

# Link nodes together
links.new(render_layers.outputs["Image"], lens_distortion.inputs["Image"])
links.new(lens_distortion.outputs["Image"], composite.inputs["Image"])

for count in range(0,NUM_IMAGES):

# Starts a new blender session (avoids appending with incrementing suffixes and other annoyances...)
    #bpy.ops.wm.read_factory_settings()

    IMG_WIDTH = 1920
    IMG_HEIGHT = 1080
    
    object_name = "Camera"

    # Get the object by name
    camera_object = bpy.data.objects.get(object_name)

    #######################################################################################################################
    #                       BG Image
    #######################################################################################################################

    material_name = "BG_material"
    BG_list    =  os.listdir(PARENT_DIR+'/BG')
    index = random.randint(0,len(BG_list)-1)
    image_path = PARENT_DIR+'/BG/'+BG_list[index]

    # Get the material
    material = bpy.data.materials.get(material_name)
    if material is None:
        print(f"Material '{material_name}' not found.")
    else:
        # Ensure the material uses nodes
        if not material.use_nodes:
            material.use_nodes = True

        # Get the node tree
        nodes = material.node_tree.nodes

        # Find an existing Image Texture node
        image_texture_node = None
        for node in nodes:
            if node.type == 'TEX_IMAGE':
                image_texture_node = node
                break

        # If no Image Texture node exists, create one
        if image_texture_node is None:
            image_texture_node = nodes.new(type='ShaderNodeTexImage')

        # Load the image
        image = bpy.data.images.load(image_path, check_existing=True)

        # Assign the image to the Image Texture node
        image_texture_node.image = image

        #######################################################################################################################
        #                       TARGET
        #######################################################################################################################

        target_location = (random.uniform(-1.0,1.0), random.uniform(10.0,14.0), random.uniform(-0.5,0.5))
        #target_location = (random.uniform(-1.0,1.0), 12.0, random.uniform(-0.5,0.5))

        #target_scale = random.uniform(0.7,1.5)

        # Get the object by name
        target_object = bpy.data.objects.get('Target')
        target_object.location = target_location

        cornerTL_object = bpy.data.objects.get('cornerTL')
        cornerTR_object = bpy.data.objects.get('cornerTR')
        cornerBL_object = bpy.data.objects.get('cornerBL')
        cornerBR_object = bpy.data.objects.get('cornerBR')

        #######################################################################################################################
        #                       RENDER OUTPUT AND INPUT
        #######################################################################################################################

        lens_distortion.inputs["Distortion"].default_value = random.uniform(-0.04,0.04)

        camera_object.data.dof.use_dof = True

        # Set focus distance (in meters)
        camera_object.data.dof.focus_distance = target_location[1]  # focus on target

        # Set f-stop (aperture)
        camera_object.data.dof.aperture_fstop = 10.0  # Excellent f-stop value

        bpy.data.scenes[0].render.filepath = PARENT_DIR+'/Lens_Dataset/OUTPUTS/'+str(count)+'.png'

        # Render OUTPUT
        bpy.ops.render.render(write_still=True)

        #----------------------------------------------------------------------------------------------------

        # Set focus distance (in meters)
        if random.random() > 0.5:
            camera_object.data.dof.focus_distance = target_location[1]+random.uniform(-4.0,-1.0)
        else:
            camera_object.data.dof.focus_distance = target_location[1]+random.uniform(1.0,4.0)

        # Set f-stop (aperture)
        camera_object.data.dof.aperture_fstop = random.uniform(0.3,2.0)  # Change to your desired f-stop value

        bpy.data.scenes[0].render.filepath = PARENT_DIR+'/Lens_Dataset/INPUTS/'+str(count)+'.png'

        # Render INPUT
        bpy.ops.render.render(write_still=True)

        cornerTL_x = str(bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera_object, cornerTL_object.matrix_world.translation).x*IMG_WIDTH)
        cornerTL_y = str(IMG_HEIGHT-bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera_object, cornerTL_object.matrix_world.translation).y*IMG_HEIGHT)
        cornerTR_x = str(bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera_object, cornerTR_object.matrix_world.translation).x*IMG_WIDTH)
        cornerTR_y = str(IMG_HEIGHT-bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera_object, cornerTR_object.matrix_world.translation).y*IMG_HEIGHT)
        cornerBL_x = str(bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera_object, cornerBL_object.matrix_world.translation).x*IMG_WIDTH)
        cornerBL_y = str(IMG_HEIGHT-bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera_object, cornerBL_object.matrix_world.translation).y*IMG_HEIGHT)
        cornerBR_x = str(bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera_object, cornerBR_object.matrix_world.translation).x*IMG_WIDTH)
        cornerBR_y = str(IMG_HEIGHT-bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera_object, cornerBR_object.matrix_world.translation).y*IMG_HEIGHT)

        with open(PARENT_DIR+'/Lens_Dataset/data.csv', 'a') as myfile:
            csv_line = str(count)+','+cornerTL_x+','+cornerTL_y+','+cornerTR_x+','+cornerTR_y+','+cornerBL_x+','+cornerBL_y+','+cornerBR_x+','+cornerBR_y+'\n'
            myfile.write(csv_line)

        os.system('cls')
        print('Generated ' + str(count+1) + ' / ' + str(NUM_IMAGES))