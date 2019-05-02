function resetHand(modelBase)
    -- See e.g. http://www.forum.coppeliarobotics.com/viewtopic.php?t=6700
    local allObjectsToExplore = {modelBase}
    while (#allObjectsToExplore > 0) do
       local obj = allObjectsToExplore[1]
       table.remove(allObjectsToExplore, 1)
       simResetDynamicObject(obj)
       local index = 0
       while true do
          child = simGetObjectChild(obj, index)
          if (child == -1) then
             break
          end
          table.insert(allObjectsToExplore, child)
          index = index+1
       end
    end
end


setGripperPose = function(inInts, inFloats, inStrings, inBuffer)
    local reset_config = inInts[1]

    local pose = {inFloats[1], inFloats[2], inFloats[3], inFloats[4],
                  inFloats[5], inFloats[6], inFloats[7], inFloats[8],
                  inFloats[9], inFloats[10], inFloats[11], inFloats[12]}

    if inFloats[12] < 0.027 then
        print("gripper pose too low")
        return {1}, {}, {}, ''
        -- pose = {inFloats[1], inFloats[2], inFloats[3], inFloats[4],
        --               inFloats[5], inFloats[6], inFloats[7], inFloats[8],
        --               inFloats[9], inFloats[10], inFloats[11], 0.027}
    end

    local h_gripper_base = simGetIntegerSignal('h_gripper_base')
    local h_gripper_dummy = simGetIntegerSignal('h_gripper_dummy')
    local h_gripper_config_buffer = simGetIntegerSignal('h_gripper_config_buffer')
    local h_object = simGetObjectHandle('object')
  
    if reset_config == 1 then
        simSetConfigurationTree(h_gripper_config_buffer)
    end
    
    resetHand(h_gripper_base)

    simSetObjectMatrix(h_gripper_dummy, -1, pose)

    resetHand(h_gripper_base)

    local h_floor = simGetObjectHandle('ResizableFloor_5_25_element')
    local h_finger_r = simGetObjectHandle('BaxterGripper_rightFinger_visible')
    local h_finger_l = simGetObjectHandle('BaxterGripper_leftFinger_visible')

    collisions_object = simCheckCollision(h_gripper_base, h_object)
    collisions_r_object = simCheckCollision(h_finger_r, h_object)
    collisions_l_object = simCheckCollision(h_finger_l, h_object)

    collisions_object = collisions_object + collisions_r_object + collisions_l_object

    if collisions_object > 0 then
        return {1}, {}, {}, ''
    end

    return {0}, {}, {}, ''
end

setGraspTarget = function(inInts, inFloats, inStrings, inBuffer)
    local reset_config = inInts[1]

    local pose = {inFloats[1], inFloats[2], inFloats[3], inFloats[4],
                  inFloats[5], inFloats[6], inFloats[7], inFloats[8],
                  inFloats[9], inFloats[10], inFloats[11], inFloats[12]}

    -- if inFloats[12] < 0.01 then
    --     return {1}, {}, {}, ''
    -- end

    local h_target = simGetObjectHandle('Sawyer_target')
    simSetObjectMatrix(h_target, -1, pose)
    print("target position set")

    return {0}, {}, {}, ''
end

setCameraPose = function(inInts, inFloats, inStrings, inBuffer)
    local reset_config = inInts[1]

    local pose = {inFloats[1], inFloats[2], inFloats[3], inFloats[4],
                  inFloats[5], inFloats[6], inFloats[7], inFloats[8],
                  inFloats[9], inFloats[10], inFloats[11], inFloats[12]}

    local h_camera_dummy = simGetIntegerSignal('h_camera_dummy')
    ---local h_camera_rgb = simGetIntegerSignal('h_camera_rgb')
    ---local h_camera_depth = simGetIntegerSignal('h_camera_depth')

    simSetObjectMatrix(h_camera_dummy, -1, pose)
    ---simSetObjectMatrix(h_camera_rgb, -1, pose)
    ---simSetObjectMatrix(h_camera_depth, -1, pose)

    return {0}, {}, {}, ''
end

setCameraPoseFromObjPose = function(inInts, inFloats, inStrings, inBuffer)
    local reset_config = inInts[1]

    local pose = {inFloats[1], inFloats[2], inFloats[3], inFloats[4],
                  inFloats[5], inFloats[6], inFloats[7], inFloats[8],
                  inFloats[9], inFloats[10], inFloats[11], inFloats[12]}

    local h_camera_dummy = simGetIntegerSignal('h_camera_dummy')
    local h_object = simGetIntegerSignal('h_object')
    ---local h_camera_rgb = simGetIntegerSignal('h_camera_rgb')
    ---local h_camera_depth = simGetIntegerSignal('h_camera_depth')

    simSetObjectMatrix(h_camera_dummy, -1, pose)

    -- local matrix = simGetObjectPosition(h_camera_dummy, -1)
    ---simSetObjectMatrix(h_camera_rgb, -1, pose)
    ---simSetObjectMatrix(h_camera_depth, -1, pose)

    return {0}, {}, {}, ''
end

setPoseByName = function(inInts, inFloats, inStrings, inBuffer)

    local pose = {inFloats[1], inFloats[2], inFloats[3], inFloats[4],
                  inFloats[5], inFloats[6], inFloats[7], inFloats[8],
                  inFloats[9], inFloats[10], inFloats[11], inFloats[12]}

    local h_part = simGetObjectHandle(inStrings[1])

    -- simSetObjectInt32Parameter(h_part, sim_shapeintparam_static, 1)
    simSetObjectInt32Parameter(h_part, sim_shapeintparam_respondable, 0)

    simSetObjectMatrix(h_part, -1, pose)

    -- simSetObjectInt32Parameter(h_part, sim_shapeintparam_static, 1)
    simSetObjectInt32Parameter(h_part, sim_shapeintparam_respondable, 1)

    simResetDynamicObject(h_part)
    return {}, {}, {}, ''
end


getPoseByName = function(inInts, inFloats, inStrings, inBuffer)
    local name = inStrings[1]
    local h_part = simGetObjectHandle(name)
    local matrix = simGetObjectMatrix(h_part, -1)

    return {}, matrix, {}, ''
end


setGripperProperties = function(inInts, inFloats, inStrings, inBuffer)

    local model_properties = {sim_modelproperty_not_collidable,
                              sim_modelproperty_not_measurable,
                              sim_modelproperty_not_renderable,
                              sim_modelproperty_not_detectable,
                              sim_modelproperty_not_cuttable,
                              sim_modelproperty_not_dynamic,
                              sim_modelproperty_not_respondable,
                              sim_modelproperty_not_visible}

    local h_gripper_base = simGetIntegerSignal('h_gripper_base')

    if #inInts ~= #model_properties then
        print('Number of model properties != # input properties.')
        print('setGripperProperties requires the following parameters in order:')
        print('sim_modelproperty_not_collidable')
        print('sim_modelproperty_not_measurable')
        print('sim_modelproperty_not_renderable')
        print('sim_modelproperty_not_detectable')
        print('sim_modelproperty_not_cuttable')
        print('sim_modelproperty_not_dynamic')
        print('sim_modelproperty_not_respondable')
        print('sim_modelproperty_not_visible')
    else
        local props = 0
        for i = 1, #inInts, 1 do
            if inInts[i] == 1 then
                props = props + model_properties[i]
            end
        end
        simSetModelProperty(h_gripper_base, props)
    end
    resetHand(h_gripper_base)

    return {}, {}, {}, ''
end

loadObject = function(inInts, inFloats, inStrings, inBuffer)
    local file_format = inInts[1]

    local use_convex_as_respondable = inInts[2]

    local mesh_path = inStrings[1]

    local com = {inFloats[1], inFloats[2], inFloats[3]}
    local mass = inFloats[4]
    local inertia = {inFloats[5],  inFloats[6],  inFloats[7],
                     inFloats[8],  inFloats[9],  inFloats[10],
                     inFloats[11], inFloats[12], inFloats[13]}

    -- Load the object and set pose, if we're interested in a new object.
    -- There seems to be an issue with memory and simCreateMeshShape when we
    -- try to delete it from the scene. So for now we'll only load when we
    -- need to.

    local h_object = simGetIntegerSignal('h_object')

    -- If we already have a mesh object in the scene, remove it
    if h_object ~= nil then
        local all = simGetObjectsInTree(h_object)
        for i = 1, #all, 1 do
            print('removing: ', simGetObjectName(all[i]))
            simRemoveObject(all[i])
        end
        simClearIntegerSignal('h_object')
    end

    -- First need to try and load the mesh (may contain many components), and
    -- then try and create it. If this doesn't work, quit the sim
    vertices, indices, _, _ = simImportMesh(file_format, mesh_path, 0, 0.001, 1.0)
    --h_object = simImportShape(file_format, mesh_path, 0, 0.001, 1.0)

    h_object = simCreateMeshShape(0, 0, vertices[1], indices[1])
    if h_object == nil then
        print('ERROR: UNABLE TO CREATE MESH SHAPE')
        simStopSimulation()
    end
    simComputeMassAndInertia(h_object, 10)

    -- Sometimes, meshes may be complex and dynamics are tricky to emulate.
    -- Here, we can calculate a convex hull for the object, and perform all
    -- grasps relative to that instead.
    --[[[
    if use_convex_as_respondable == 1 then
        local vert, idx = simGetQHull(vertices[1])
        local h_object_2 = simCreateMeshShape(0, 0, vert, idx)

        simSetObjectMatrix(h_object, h_object_2, {0,0,0,0,0,0,0,0,0,0,0,0})
        simSetObjectInt32Parameter(h_object_2, sim_objintparam_visibility_layer, 0)
        simSetObjectParent(h_object, h_object_2, true)
        simSetModelProperty(h_object, sim_modelproperty_not_collidable +
                                      sim_modelproperty_not_measurable +
                                      sim_modelproperty_not_dynamic +
                                      sim_modelproperty_not_respondable,
                                      sim_modelproperty_not_detectable)
        simSetObjectSpecialProperty(h_object, sim_objectspecialproperty_renderable)
        h_object = h_object_2
    end
    --]]
    simSetIntegerSignal('h_object', h_object)

    simSetObjectName(h_object, 'object')
    simSetObjectInt32Parameter(h_object, sim_shapeintparam_respondable, 1)
    simSetObjectInt32Parameter(h_object, sim_shapeintparam_static, 1)
    simReorientShapeBoundingBox(h_object, -1)

    ---simSetModelProperty(h_object, 0)

    simSetShapeMaterial(h_object, simGetMaterialId('usr_sticky'))

    --- By default, the absolute reference frame is used. We re-orient the
    -- object to be WRT this frame by default, so don't need an extra mtx.
    ---simSetShapeMassAndInertia(h_object, mass, inertia, com)

    -- Playing with ODE & vortex engines
    ---local frictionVortex = simGetEngineFloatParameter(sim_vortex_body_primlinearaxisfriction, objectHandle)
    ---simSetEngineFloatParameter(sim_ode_body_friction, h_object, 0.9)
    ---simResetDynamicObject(h_object)

    simSetEngineFloatParameter(sim_vortex_body_skinthickness, h_object, 0.05)
    simSetEngineFloatParameter(sim_vortex_body_primlinearaxisfriction, h_object, 0.75)
    simSetEngineFloatParameter(sim_vortex_body_seclinearaxisfriction, h_object, 0.75)
    ---simSetEngineBoolParameter(sim_vortex_body_randomshapesasterrain, h_object, true)
    simSetEngineBoolParameter(sim_vortex_body_autoslip, h_object, true)

    sim.setShapeColor(h_object, nil, sim_colorcomponent_ambient_diffuse, {1,0,0})

    simResetDynamicObject(h_object)

    return {h_object}, {}, {}, ''
end

createObject = function(inInts, inFloats, inStrings, inBuffer)

    local mesh_path = inStrings[1]

    -- If we already have a mesh object in the scene, remove it
    local h_object = simGetIntegerSignal('h_object')
    if h_object ~= nil then
        local all = simGetObjectsInTree(h_object)
        for i = 1, #all, 1 do
            print('removing: ', simGetObjectName(all[i]))
            simRemoveObject(all[i])
        end
        simClearIntegerSignal('h_object')
    end

    if h_object == nil then
        print('ERROR: UNABLE TO CREATE MESH SHAPE')
        simStopSimulation()
    end

    n = math.random(1, 5)
    shapes = {}

    for n_i = 0, n, 1 do
        x = 2 + math.random() * (2)
        y = 2 + math.random() * (2)
        z = 2 + math.random() * (2)
        h_object = simCreatePureShape(0,0,{x,y,z},1,nil)
        shapes[n_i] = h_object

        if h_object == nil then
            print('ERROR: UNABLE TO CREATE MESH SHAPE')
            simStopSimulation()
        end

        simSetObjectMatrix(h_gripper_dummy, -1, pose)
    end
    h_object = simGroupShapes(shapes, n)

    simSetIntegerSignal('h_object', h_object)

    simSetObjectName(h_object, 'object')

    simResetDynamicObject(h_object)
    
    allVertices={}
    allIndices={}
    allNames={}
    h_object_handle = simGetObjectHandle('object')
    --print(h_object_handle)
    --h = simGetObjects(h_object_handle,sim.object_shape_type)
    --print(h)
    vertices,indices = simGetShapeMesh(h_object_handle)
    m = simGetObjectMatrix(h_object_handle,-1)

    for i=1,#vertices/3,1 do
        v={vertices[3*(i-1)+1],vertices[3*(i-1)+2],vertices[3*(i-1)+3]}
        v=sim.multiplyVector(m,v)
        vertices[3*(i-1)+1]=v[1]
        vertices[3*(i-1)+2]=v[2]
        vertices[3*(i-1)+3]=v[3]
    end

    table.insert(allVertices,vertices)
    table.insert(allIndices,indices)
    table.insert(allNames,sim.getObjectName(h_object_handle))

    if (#allVertices>0) then
        ---simExportMesh(0,"/../../dex-net/generated_shapes/example.obj",0,1,allVertices,allIndices,nil,allNames)
        simExportMesh(0,mesh_path,0,1,allVertices,allIndices,nil,allNames)
    end

    return {h_object}, {}, {}, ''
end

saveObject = function(inInts, inFloats, inStrings, inBuffer)
    local mesh_path = inStrings[1]
    print(mesh_path)
    local obj_name = inStrings[2]

    print(obj_name)
    h_object_handle = simGetObjectHandle(obj_name)
    
    allVertices={}
    allIndices={}
    allNames={}
    vertices,indices = simGetShapeMesh(h_object_handle)
    m = simGetObjectMatrix(h_object_handle,-1)

    for i=1,#vertices/3,1 do
        v={vertices[3*(i-1)+1],vertices[3*(i-1)+2],vertices[3*(i-1)+3]}
        v=sim.multiplyVector(m,v)
        vertices[3*(i-1)+1]=v[1]
        vertices[3*(i-1)+2]=v[2]
        vertices[3*(i-1)+3]=v[3]
    end

    table.insert(allVertices,vertices)
    table.insert(allIndices,indices)
    table.insert(allNames,sim.getObjectName(h_object_handle))

    if (#allVertices>0) then
        ---simExportMesh(0,"/../../dex-net/generated_shapes/example.obj",0,1,allVertices,allIndices,nil,allNames)
        simExportMesh(0,mesh_path,0,1,allVertices,allIndices,nil,allNames)
    end

    return {h_object}, {}, {}, ''

end


--- Called from a remote client, and returns rendered images of a scene.
-- This function loads an object, sets its pose, and performs a small set of
-- domain randomization to get different augmentations of the scene.
queryCamera = function(inInts, inFloats, inStrings, inBuffer)

    local res = inInts[1] -- resolution
    local randomize_texture = inInts[2]
    local randomize_colour = inInts[3]
    local randomize_lighting = inInts[4]


    local work2cam = {
        inFloats[1], inFloats[2],  inFloats[3],  inFloats[4],
        inFloats[5], inFloats[6],  inFloats[7],  inFloats[8],
        inFloats[9], inFloats[10], inFloats[11], inFloats[12]}

    local p_light_off = inFloats[13]
    local p_light_mag = inFloats[14]
    local rgbNearClip = inFloats[15]
    local rgbFarClip = inFloats[16]
    local depthNearClip = inFloats[17]
    local depthFarClip = inFloats[18]
    local fov = inFloats[19]

    local texturePath = inStrings[1]


    -- Get object handles
    local h_object = simGetIntegerSignal('h_object')
    local h_workspace = simGetIntegerSignal('h_workspace')
    local h_camera_rgb = simGetIntegerSignal('h_camera_rgb')
    local h_camera_depth = simGetIntegerSignal('h_camera_depth')
    ---local h_lights = simGetStringSignal('h_lights')
    ---local light_default_pos = simGetStringSignal('light_default_pos')
    ---local light_default_ori = simGetStringSignal('light_default_ori')

    ---h_lights = simUnpackInt32Table(h_lights)
    ---light_default_pos = simUnpackTable(light_default_pos)
    ---light_default_ori = simUnpackTable(light_default_ori)

    ---if randomize_texture == 1 then
    ---    randomizeTexture(h_object, texturePath)
    ---    randomizeTexture(h_table_object, texturePath)
    ---end
    ---if randomize_colour == 1 then
    ---    randomizeColour(h_object)
    ---    randomizeColour(h_table_object)
    ---end
    ---if randomize_lighting == 1 then
    ---    randomizeLighting(h_lights, light_default_pos,
    ---                      light_default_ori, p_light_mag, p_light_off)
    ---end


    -- Set the resolution for each camera
    simSetObjectFloatParameter(h_camera_rgb, sim_visionfloatparam_near_clipping, rgbNearClip)
    simSetObjectFloatParameter(h_camera_rgb, sim_visionfloatparam_far_clipping, rgbFarClip)
    simSetObjectFloatParameter(h_camera_depth, sim_visionfloatparam_near_clipping, depthNearClip)
    simSetObjectFloatParameter(h_camera_depth, sim_visionfloatparam_far_clipping, depthFarClip)

    -- Set Field of View (fov), resolution, and object to visualize
    for _, cam in pairs({h_camera_depth, h_camera_rgb}) do
        simSetObjectFloatParameter(cam, sim_visionfloatparam_perspective_angle, fov)
        simSetObjectInt32Parameter(cam, sim_visionintparam_resolution_x, res)
        simSetObjectInt32Parameter(cam, sim_visionintparam_resolution_y, res)
        ---simSetObjectMatrix(cam,-1, work2cam)

        -- Allow camera to capture all renderable objects in scene
        simSetObjectInt32Parameter(cam, sim_visionintparam_entity_to_render, -1)
    end

    --- We only need a single picture of the object, so we need to
    -- make sure that the simulation knows to render it now
    simHandleVisionSensor(h_camera_depth)
    simHandleVisionSensor(h_camera_rgb)

    local depth_image = simGetVisionSensorDepthBuffer(h_camera_depth)
    local colour_image = simGetVisionSensorImage(h_camera_rgb)

    local all_images = table.copy({}, depth_image,  colour_image)

    return {}, all_images, {}, ''
end

setCameraResolution = function(inInts, inFloats, inStrings, inBuffer)
    local h_camera_rgb = simGetIntegerSignal('h_camera_rgb')
    local h_camera_depth = simGetIntegerSignal('h_camera_depth')

    local resX = inInts[1]
    local resY = inInts[2]

    simSetObjectInt32Parameter(h_camera_rgb, sim_visionintparam_resolution_x, resX)
    simSetObjectInt32Parameter(h_camera_rgb, sim_visionintparam_resolution_y, resY)
    simSetObjectInt32Parameter(h_camera_depth, sim_visionintparam_resolution_x, resX)
    simSetObjectInt32Parameter(h_camera_depth, sim_visionintparam_resolution_y, resY)

    return {}, {}, {}, ''
end

if (sim_call_type == sim_childscriptcall_initialization) then

    simClearStringSignal('grasp_candidate')
    simClearStringSignal('drop_object')

    -- ----------------- SIMULATION OBJECT HANDLES ----------------------------
    --- All object handles will be prefixed with a 'h_' marker.
    --- All handles to be used in the simulation will be collected here, and
    -- stored as "global" variables. Only this local function will have write
    -- access however.
    --- Functions called by a remote client will have access to the global
    -- variables, but all subroutines called by those functions will need to
    -- have these parameters passed in as values.

    --local h_gripper_base = simGetObjectHandle('ROBOTIQ_85')
    --local h_gripper_base = simGetObjectHandle('BarrettHand')
    local h_gripper_base = simGetObjectHandle('BaxterGripper')

    local h_camera_rgb = simGetObjectHandle('Vision_sensor_rgb')
    local h_camera_depth = simGetObjectHandle('Vision_sensor_depth')
    local h_camera_dummy = simGetObjectHandle('camera_dummy')

    simSetIntegerSignal('h_camera_rgb', h_camera_rgb)
    simSetIntegerSignal('h_camera_depth', h_camera_depth)
    simSetIntegerSignal('h_camera_dummy', h_camera_dummy)



    local h_gripper_dummy = simGetObjectHandle('gripper_dummy')
    local h_gripper_config_buffer = simGetConfigurationTree(h_gripper_base)


    --- Given the name of the root of the gripper model, we traverse through all
    -- components to find the contact points.
    local h_gripper_contacts = {}
    local h_gripper_respondable = {}
    local h_gripper_joints = {}

    -- ------------------- VISUALIZATION HANDLES ---------------------------

    --- In V-REP, there are a few useful modules for drawing to screen, which
    -- makes things like debugging much easier.
    local display_num_points = 5000;
    local display_point_density = 0.001;
    local display_point_size = 0.005;
    local display_vector_width = 0.001;
    local black = {0, 0, 0}; local purple = {1, 0, 1};
    local blue = {0, 0, 1}; local red = {1, 0, 0}; local green = {0, 1, 0};

    -- ------------- MISC GRIPPER / MODEL PARAMETERS --------------------

    local num_collision_thresh = 75

    -- Used for lifting the object
    local max_vel_accel_jerk = {0.2, 0.2, 0.2,
                                0.05, 0.05, 0.05,
                                0.2, 0.2, 0.2};

    gripper_prop_static =
        sim_modelproperty_not_dynamic    +
        sim_modelproperty_not_renderable +
        sim_modelproperty_not_collidable  +
        sim_modelproperty_not_respondable

    gripper_prop_visible =
        sim_modelproperty_not_renderable +
        sim_modelproperty_not_measurable

    gripper_prop_invisible =
        sim_modelproperty_not_collidable  +
        sim_modelproperty_not_renderable  +
        sim_modelproperty_not_visible     +
        sim_modelproperty_not_respondable +
        sim_modelproperty_not_dynamic

    simSetModelProperty(h_gripper_base, gripper_prop_invisible)

    -- ---------------- HANDLES FOR SCENE RANDOMIZATION -----------------------

    -- Finally, always start the lights at these default positions
    local light_default_pos = {{2.685, 0.4237, 4.2},
                               {2.5356, 0.7, 4.256},
                               {2.44, 0.4022, 4.3476},
                               {2.44, 0.402, 4.3477}}

    -- These are the default orientations of the cameras, when reading off of
    -- the screen, but future functions (i.e. simSetObjectOrientation) expects
    -- input angles to be in radians.
    local light_default_ori = {{-122.7, -64, 130.6},
                               {102.8, 1.897, 19.80},
                               {-106.75, 52.73, -145.27},
                               {-180, 0, 0}}

    for i, light in ipairs(light_default_ori) do
        light_default_ori[i] = {(3.1415 * light[1] / 180.),
                                (3.1415 * light[2] / 180.),
                                (3.1415 * light[3] / 180.)}
    end


    -- ------- SAVE HANDLES AS GLOBAL VARIABLES (i.e. as signals) -------------

    simSetIntegerSignal('h_gripper_dummy', h_gripper_dummy)
    simSetIntegerSignal('h_gripper_base', h_gripper_base)
    simSetIntegerSignal('h_gripper_config_buffer', h_gripper_config_buffer)

    -- Check where the data will come from
    local PORT_NUM = simGetStringParameter(sim_stringparam_app_arg1)

    if PORT_NUM == '' then
        PORT_NUM = 19999 -- default
        simExtRemoteApiStart(PORT_NUM)
    end
    print('port num: ', PORT_NUM)
end
