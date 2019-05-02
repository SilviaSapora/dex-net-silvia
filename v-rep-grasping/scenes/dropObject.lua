threadDropFunction=function()

    simWaitForSignal('h_object')

    local h_object = simGetIntegerSignal('h_object')
    local h_table_object = simGetIntegerSignal('h_table_object')

    while simGetSimulationState() ~= sim_simulation_advancing_abouttostop do

        local object_pose = simGetStringSignal('run_drop_object')
        if object_pose and #object_pose > 0 then

            object_pose = simUnpackFloatTable(object_pose)
            simClearStringSignal('run_drop_object')

            h_object = simGetIntegerSignal('h_object')

            -- Set the object pose & make respondable
            simSetObjectMatrix(h_object, -1, object_pose)
            -- simSetObjectInt32Parameter(h_object, sim_shapeintparam_static, 1)
            -- simSetObjectInt32Parameter(h_object, sim_shapeintparam_respondable, 0)
            -- simSwitchThread()

            --- Set the object static so it doesn't move before contact
            ---simSetObjectInt32Parameter(h_object, sim_shapeintparam_static, 1)
            ---simResetDynamicObject(h_object)
            ---simSwitchThread()

             -- Move object to center of workspace (keep orientation)
            --local obj_pos = simGetObjectPosition(h_object, -1)

            -- simSetObjectPosition(h_object, -1, {0, 0, 0.001})

            simSetObjectInt32Parameter(h_object, sim_shapeintparam_respondable, 1)
            simSetObjectInt32Parameter(h_object, sim_shapeintparam_static, 0)
            simResetDynamicObject(h_object)
            
            simSetIntegerSignal('object_resting', 1)
        end
        simSwitchThread()
    end
end

-- Here we execute the regular thread code:
res,err=xpcall(threadDropFunction, function(err) return debug.traceback(err) end)
if not res then
    print('Error: ', err)
    simAddStatusbarMessage('Lua runtime error: '..err)
end
