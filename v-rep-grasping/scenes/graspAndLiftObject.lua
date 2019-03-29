--math.randomseed(os.time())
math.randomseed(1234)

-- MAIN FUNCTION
threadCollectionFunction = function()

    -- For initialization from remoteApiCommandServer
    simWaitForSignal('h_object')

    --[[local h_object = simGetIntegerSignal('h_object')
    local h_workspace = simGetIntegerSignal('h_workspace')
    local h_table_object = simGetIntegerSignal('h_table_object')
    local h_gripper_palm = simGetIntegerSignal('h_gripper_palm')
    local h_gripper_dummy = simGetIntegerSignal('h_gripper_dummy')
    local h_gripper_prox = simGetIntegerSignal('h_gripper_prox')
    local max_vel_accel_jerk = simGetStringSignal('max_vel_accel_jerk')
    --]]
    --max_vel_accel_jerk = simUnpackFloatTable(max_vel_accel_jerk)
    local max_vel_accel_jerk = {0.004, 0.004, 0.004,
                                0.001, 0.001, 0.001,
                                0.17, 0.17, 0.17};

    local h_gripper_base = simGetIntegerSignal('h_gripper_base')
    local h_floor = simGetObjectHandle('ResizableFloor_5_25_element')

    while simGetSimulationState() ~= sim_simulation_advancing_abouttostop do

        local finger_angle = simGetIntegerSignal('run_grasp_attempt')
        if finger_angle ~= nil then

            simClearIntegerSignal('run_grasp_attempt')
            ---simSetScriptSimulationParameter(sim_handle_all, 'fingerAngle', finger_angle)

            --- These will save the results of the grasp attempt; a successful
            -- or failed grasp will be encoded here; otherwise we return
            -- default values
            h_object = simGetIntegerSignal('h_object')
            h_gripper_dummy = simGetIntegerSignal('h_gripper_dummy')

            simSetObjectInt32Parameter(h_object, sim_shapeintparam_static, 0)
            print('run grasp attempt')

            ----- LIFT THE OBJECT -----
            -- Make object dynamic
            simResetDynamicObject(h_object)
            simSwitchThread()

            -- ---------------- GRASP THE OBJECT ---------------------
            simSetIntegerSignal('closeGrasp', 1)
            simWaitForSignal('grasp_done')
            simClearIntegerSignal('grasp_done')
            simClearIntegerSignal('closeGrasp')

            simWait(1)

            -- Send a signal to hold the grasp while we attempt a lift
            --simSetIntegerSignal('holdGrasp', 1)
            --simWaitForSignal('grasp_done')
            --simClearIntegerSignal('grasp_done')
            --simSwitchThread()

            --- Define a path that leads from the current gripper
            -- position to a final gripper position. To perform
            -- this 'lift' action, we manually follow the path by
            -- manually setting the new position at each time step.

            local p0 = simGetObjectPosition(h_gripper_dummy, -1)
            local targetPosVel = {0, 0, 0.2, 0, 0, 0}
            local posVelAccel = {p0[1], p0[2], p0[3], 0, 0, 0, 0, 0, 0, 0}

            local rmlHandle = simRMLPos(3, 0.001, -1, posVelAccel,
                                        max_vel_accel_jerk,
                                        {1,1,1}, targetPosVel)

            -- Incrementally move the hand along the trajectory
            local res = 0
            print('moving along trajectory')
            while res == 0 do
                dt = simGetSimulationTimeStep()
                res, posVelAccel, sync = simRMLStep(rmlHandle,dt)
                simSetObjectPosition(h_gripper_dummy, -1, posVelAccel)
            end
            simRMLRemove(rmlHandle)

            object_on_floor = sim.checkCollision(h_object, h_floor)
            ----- END LIFT THE OBJECT -----

            -- Let the object fall onto table
            simSetIntegerSignal('clearGrasp', 1)
            simWaitForSignal('grasp_done')
            simClearIntegerSignal('grasp_done')

            --- Set the object static so it doesn't move before contact
            ---simSetObjectInt32Parameter(h_object, sim_shapeintparam_static, 1)
            simResetDynamicObject(h_object)
            print('set signal done')
            simSetStringSignal('py_grasp_done', object_on_floor)

        end  -- Check gripper collisions

        ---simSwitchThread()
    end -- loop infinite
end

simSetBooleanParameter(sim_boolparam_dynamics_handling_enabled, true)

-- Here we execute the regular thread code:
res,err=xpcall(threadCollectionFunction, function(err) return debug.traceback(err) end)
if not res then
    print('Error: ', err)
    simAddStatusbarMessage('Lua runtime error: '..err)
end
print('Done simulation!')
