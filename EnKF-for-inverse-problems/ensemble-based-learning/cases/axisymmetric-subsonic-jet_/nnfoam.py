# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for OpenFOAM simpleFoam solver with kOmegaQuadratic turbulence model """

# standard library imports
import os
import shutil
import subprocess
import multiprocessing

# third party imports
import numpy as np
import scipy.sparse as sp
import scipy
import yaml

# local imports
from dafi import PhysicsModel
from dafi import random_field as rf
from dafi.random_field import foam

import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

import neuralnet
import gradient_descent as gd
import regularization as reg
import data_preproc as preproc
import cost
from get_inputs import get_inputs_loc_norm


import pdb

TENSORDIM = 9
TENSORSQRTDIM = 3
DEVSYMTENSORDIM = 5                     # deviatoric part of Reynolds stress
DEVSYMTENSOR_INDEX = [0,1,2,4,5]
NBASISTENSORS = 10                      # num of T: tensor basis of Reynolds stress's deviatoric part
NSCALARINVARIANTS = 5                   # num of theta: scalar invariants

VECTORDIM = 3

# for centerline
C_x = 119
stp_x=np.array([7, 5])

class Model(PhysicsModel):

    def __init__(self, inputs_dafi, inputs_model):

        """ the dafi.in is seperated into 3 parts
                dafi    - inputs_dafi,
                inverse - inputs_inverse,
                model   - inputs_model. """

        # get required DAFI inputs.
        self.nsamples = inputs_dafi['nsamples']
        max_iterations = inputs_dafi['max_iterations']
        self.analysis_to_obs = inputs_dafi['analysis_to_obs']

        # read input file
        self.foam_case = inputs_model['foam_case']
        iteration_nstep = inputs_model['iteration_nstep']           # CFD simulation: timeStep = 1 / iteration_nstep
        self.foam_timedir = str(iteration_nstep)

        self.ncpu = inputs_model.get('ncpu', 20)
        self.rel_stddev = inputs_model.get('rel_stddev', 0.5)
        self.abs_stddev = inputs_model.get('abs_stddev', 0.5)
        self.obs_rel_std = inputs_model.get('obs_rel_std', 0.001)
        self.obs_abs_std = inputs_model.get('obs_abs_std', 0.0001)

        weight_baseline_file = inputs_model['weight_baseline_file']

        # required attributes
        self.name = 'NN parameterized RANS model'

        # results directory
        self.results_dir = 'results_ensemble'

        # counter
        self.da_iteration = -1

        iteration_step_length = 1.0 / iteration_nstep

        # control dictionary
        self.writeprecision = 6
        self.control_list = {
            'application': 'simpleFoam',
            'startFrom': 'latestTime',
            'startTime': '0',
            'stopAt': 'nextWrite',
            'endTime': f'{max_iterations}',                         
            'deltaT': f'{iteration_step_length}',
            'writeControl': 'runTime',
            'writeInterval': '1',
            'purgeWrite': '2',
            'writeFormat': 'ascii',
            'writePrecision': f'{self.writeprecision}',
            'writeCompression': 'off',
            'timeFormat': 'fixed',
            'timePrecision': '0',
            'runTimeModifiable': 'true',
        }

        nut_base_foamfile = inputs_model['nut_base_foamfile']
        self.foam_info = foam.read_header(nut_base_foamfile)
        self.foam_info['file'] = os.path.join(
            self.foam_case,'foam_base_ASJet','system', 'controlDict')

        # NN architecture
        self.nscalar_invariants = inputs_model.get('nscalar_invariants', NSCALARINVARIANTS)
        self.nbasis_tensors = inputs_model.get('nbasis_tensors', NBASISTENSORS)

        nhlayers = inputs_model.get('nhlayers', 10)
        nnodes = inputs_model.get('nnodes', 10)
        alpha = inputs_model.get('alpha', 0.0)

        # initial weights
        self.w_init = np.loadtxt(weight_baseline_file)
        self.nbasis = self.nbasis_tensors
        self.nstate = len(self.w_init)        

        # initial g
        self.g_init  = np.array(inputs_model.get('g_init', [0.0]*self.nbasis))
        self.g_scale = inputs_model.get('g_scale', 1.0)

        # initial beta
        self.beta_init = np.array(inputs_model.get("beta_init", [0.0]))
        self.beta_scale = inputs_model.get("beta_scale", 1.0)

        # data pre-processing
        self.preproc_class = inputs_model.get('preproc_class', None)

        parallel = inputs_model.get('parallel', True)

        ## CREATE NN
        self.nn = neuralnet.NN(self.nscalar_invariants, self.nbasis_tensors,
            nhlayers, nnodes, alpha)

        # call Tensorflow to get initialization messages out of the way
        with tf.GradientTape(persistent=True) as tape:
            gtmp = self.nn(np.zeros([1, self.nscalar_invariants]))
        _ = tape.jacobian(gtmp, self.nn.trainable_variables, experimental_use_pfor=False)

        self.g_data_list1=[]
        for ibasis in range(self.nbasis):
            g_file = os.path.join(self.foam_case, 'foam_base_ASJet', '0', f'g{ibasis+1}')
            g_data = rf.foam.read_field_file(g_file)
            g_data['file'] = os.path.join(self.foam_case, 'foam_base_ASJet', '0', f'g{ibasis+1}')
            self.g_data_list1.append(g_data)
        self.ncell1 = len(g_data['internal_field']['value'])

        self.w_shapes = neuralnet.weights_shape(self.nn.trainable_variables)

        # print NN summary
        print('\n' + '#'*80 + '\nCreated NN:' +
            f'\n  Number of scalar invariants: {self.nscalar_invariants}' +
            f'\n  Number of basis tensors: {self.nbasis_tensors}' +
            f'\n  Number of trainable parameters: {self.nn.count_params()}' +
            '\n' + '#'*80)

        # write the controlDict for OF
        foam.write_controlDict(
            self.control_list, self.foam_info['foam_version'],
            self.foam_info['website'], ofcase=self.foam_case+'/foam_base_ASJet')

        # get the preprocesing class
        if self.preproc_class is not None:
            self.PreProc = getattr(preproc, self.preproc_class)

        # calculate inputs
        # initialize preprocessing instance
        if os.path.isdir(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)
        self.preprocess_data1 = self.PreProc()

        # observations
        # read observations: centerline
        norm_truth = 1
        prof_center = np.loadtxt('inputs/ASJet_simpleFoam_kOmega/postProcessing/sampleDict/0/' + 'NASA_center.dat')
        u_exp_center_part1 = prof_center[0:C_x:stp_x[0],2]
        u_exp_center_part2 = prof_center[C_x::stp_x[1],2]
        u_exp_center = np.hstack([u_exp_center_part1, u_exp_center_part2])

        Ux = u_exp_center

        self.Uxobs_scalex = 1.0 / max(abs(Ux))                      # scale const: for Ux
        obs_case = Ux * self.Uxobs_scalex / norm_truth

        self.obs = obs_case

        self.nstate_obs = len(obs_case)
        print('number of observation = ', self.nstate_obs)

        # create sample directories
        sample_dirs = []
        for isample in range(self.nsamples):
            sample_dir = self._sample_dir(isample)
            sample_dirs.append(sample_dir)
            shutil.copytree(self.foam_case, sample_dir)
        self.sample_dirs = sample_dirs

    def __str__(self):
        return 'Dynamic model for simpleFoam solver.'

    # required methods
    def generate_ensemble(self):
        """ 
        Return states at the first data assimilation time-step.
        Creates the OpenFOAM case directories for each sample, creates samples.
        """

        # update X: neuralnetwork weight, num of samples -> add noise to the initial values
        w = np.zeros([self.nstate, self.nsamples])
        for i in range(self.nstate):
            w[i, :] = self.w_init[i] + np.random.normal(0,
                abs(self.w_init[i] * self.rel_stddev + self.abs_stddev), self.nsamples)
        return w

    def state_to_observation(self, state_vec):
        """ 
        Map the states to observation space (from X to HX).
        Runs OpenFOAM, and returns specific values at the observation locations.
        """
        self.da_iteration += 1          # from here: da_iteration = 0 const
                                        # da_iteration: will increase as the function being called

        # set weights
        w = state_vec.copy()
        time_dir = f'{self.da_iteration:d}'
        gsamps_case1 = []

        ts = time.time()

        parallel = multiprocessing.Pool(self.nsamples)
        
        # store the dirs and corresponding samples 
        inputs = [(self._sample_dir(i), 
            self.da_iteration) for i in range(self.nsamples)]

        # parallel computation: samples' scalar invariants and tensor basis
        input_scalars_list = parallel.starmap(_input_feature, inputs)       
        parallel.close()

        t_load = time.time()
        print(f'      load data ... {t_load-ts:.2f}s')

        # search all the samples to update the "self.stats"
        for isamp in range(self.nsamples):
            # sample i: scalar invariants, a 2D array - number of mesh points * number of scalars
            input_scalars1 = input_scalars_list[isamp]

            # update the states: the max/min of scalar invariants are changed: theta_1, 2
            self.preprocess_data1.update_stats(input_scalars1)

        # MinMax Scale
        # save stats: min max - [[min0, min1], [max0, max1]]
        # save the max/min values: _n_m: 
        #       n(0 / 1): 0 - min; 1 - max
        #       m(0 ~ max_iterations): series of iterations
        for i, stat in enumerate(self.preprocess_data1.stats):
            file = os.path.join(self.results_dir,
                f'input1_preproc_stat_{i}_{self.da_iteration}')
            np.savetxt(file, np.atleast_1d(stat))

        for isamp in range(self.nsamples):
            # sample i: scalar invariants, a 2D array - number of mesh points * number of scalars
            input_scalars1 = input_scalars_list[isamp]

            # scale
            input_scalars1_scale = self.preprocess_data1.scale(
                input_scalars1, self.preprocess_data1.stats)

            w_reshape = neuralnet.reshape_weights(w[:, isamp], self.w_shapes)
            self.nn.set_weights(w_reshape)

            """
            g_scale and g_init: limit the values of g_case1 - make the OF could run stably
            """
            # evaluate NN: cost and gradient
            # ensure the value of g1 is negative: g1 = exp(out_NN * scale)*g1_init
            with tf.GradientTape(persistent=True) as tape:
                out_NN = self.nn(input_scalars1_scale)
                g_case1=np.zeros((self.ncell1, self.nbasis))
                g_case1[:, 0] = np.exp(out_NN[:, 0] * self.g_scale[0]) * self.g_init[0]
                g_case1[:, 1] = out_NN[:, 1] * self.g_scale[1] + self.g_init[1]

            gsamps_case1.append(g_case1)

        t_eva = time.time()

        print(f'      NN evaluate ... {time.time()-t_eva:.2f}s')
        print(f'      TensorFlow ... {time.time()-ts:.2f}s')

        """
        the g files update in every 'state_to_observation' run: be consts insides the OF running
        """
        # write sample
        for i in range(self.nsamples):
            ig_case1 = np.zeros(g_case1.shape)

            for j in range(self.nbasis):
                ig_case1[:, j] = gsamps_case1[i][:, j]

            # modify the ensembles' g1 g2 files with corresponding values: use their own "w"
            self._modify_foam_case(self.g_data_list1, ig_case1, self.da_iteration, foam_dir=self._sample_dir(i)+'/foam_base_ASJet')

        # run the ensembles' OpenFoam simulations
        inputs = []
        parallel = multiprocessing.Pool(self.ncpu)
        for i in range(self.nsamples):
            inputs.append((self._sample_dir(i) + '/foam_base_ASJet', self.da_iteration, self.writeprecision))

        _ = parallel.starmap(_run_foam, inputs)
        parallel.close()

        # get HX
        norm_truth = 1
        state_in_obs = np.empty([self.nstate_obs, self.nsamples])
        for isample in range(self.nsamples):

            file = os.path.join(self._sample_dir(isample), 'foam_base_ASJet',
                'postProcessing', 'sampleDict', time_dir)

            prof_center = np.loadtxt(file + '/line_center_U.xy')
            prof_jet_exit = np.loadtxt(file + '/line_jet_exit_U.xy')

            # get the average velocity at the exit of jet
            NUM=96      # ignore the BC-wall's effect
            u_jet_exit_mean=0
            for i in range(NUM-1):
                u_jet_exit_mean=u_jet_exit_mean+(prof_jet_exit[i,1]+prof_jet_exit[i+1,1])/2*(i*2+1)/(pow(NUM-1,2))

            u_OF_center_part1 = prof_center[0:C_x:stp_x[0],1]/u_jet_exit_mean
            u_OF_center_part2 = prof_center[C_x::stp_x[1],1]/u_jet_exit_mean
            u_OF_center = np.hstack([u_OF_center_part1, u_OF_center_part2])

            Ux = u_OF_center

            model_output = Ux * self.Uxobs_scalex / norm_truth

            state_in_obs[:, isample] = model_output

        return state_in_obs

    def get_obs(self, time):
        """ 
        Return the observation and error matrix.
        """
        obs = self.obs
        obs_error = np.diag(self.obs_rel_std * abs(obs) + self.obs_abs_std)
        return obs, obs_error

        # return self.obs, self.obs_error

    def clean(self, loop):
        if loop == 'iter' and self.analysis_to_obs:
            for isamp in range(self.nsamples):
                dir = os.path.join(self._sample_dir(isamp),
                                   f'{self.da_iteration + 1:d}')
                shutil.rmtree(dir)                                          # remove all the sub-folders and files

    # internal methods
    def _sample_dir(self, isample):
        "Return name of the sample's directory. "
        return os.path.join(self.results_dir, f'sample_{isample:d}')

    def _modify_foam_case(self, g_data_list, g, step, foam_dir=None):
        for i, g_data in enumerate(g_data_list):
            g_data['internal_field']['value'] = g[:, i]
            if foam_dir is not None:
                g_data['file'] = os.path.join(foam_dir, str(step), f'g{i+1}')
            _ = rf.foam.write_field_file(**g_data)


# without 'self': these functions are internal methods
def _input_feature(foam_dir, iteration):
    gradU_file = os.path.join(foam_dir, 'foam_base_ASJet', str(iteration), 'grad(U)')
    tke_file = os.path.join(foam_dir, 'foam_base_ASJet', str(iteration), 'k')
    time_scale_file = os.path.join(foam_dir, 'foam_base_ASJet', str(iteration),  'timeScale')
    gradU = rf.foam.read_tensor_field(gradU_file)
    tke = rf.foam.read_scalar_field(tke_file)
    time_scale = rf.foam.read_scalar_field(time_scale_file)      
    
    # get the scalar invariants and tensor basis
    input_scalars = get_inputs_loc_norm(gradU, time_scale)      
    input_scalars2 = input_scalars[:, :2]
    input_scalars_list = input_scalars2

    file = os.path.join(foam_dir, f'input_scalar_{iteration}')
    np.savetxt(file, input_scalars_list)

    return input_scalars_list


# Gradient: analytic dTau/dg
def _get_dadg(tensors, tke):
    tke = np.expand_dims(np.squeeze(tke), axis=(1, 2))
    return 2.0*tke*tensors

def _clean_foam(foam_dir):
    bash_command = './clean > /dev/null'
    bash_command = 'cd ' + foam_dir + ';' + bash_command + '; cd -'
    return subprocess.call(bash_command, shell=True, stdout=subprocess.DEVNULL) # run .sh command

def _run_foam(foam_dir, iteration, writeprecision):
    # run foam
    solver = 'simpleFoam'
    logfile = os.path.join(foam_dir, solver + '.log')
    bash_command = f'{solver} -case {foam_dir} > {logfile}'
    subprocess.call(bash_command, shell=True)

    logfile = os.path.join(foam_dir, 'gradU.log')
    bash_command = f"postProcess -func 'grad(U)' -case {foam_dir}" + \
         f"> {logfile}" # 2>&1'
    subprocess.call(bash_command, shell=True)

    # move results from i to i-1 directory
    tsrc = f'{iteration + 1:d}'
    src = os.path.join(foam_dir, tsrc)
    dst = os.path.join(foam_dir, f'{iteration + 1:d}')
    shutil.move(src, dst)
    for field in ['U', 'p', 'phi', 'grad(U)', 'timeScale', 'k', 'nut', 'omega', 'g1', 'g2', 'g3', 'g4']:
        src = os.path.join(foam_dir, f'{iteration + 1:d}', field)
        dst = os.path.join(foam_dir, f'{iteration:d}', field)
        shutil.copyfile(src, dst)

    logfile = os.path.join(foam_dir, 'sample.log')
    bash_command = f"postProcess -func 'sampleDict' -case {foam_dir}" + \
         f"> {logfile}"
    subprocess.call(bash_command, shell=True)
