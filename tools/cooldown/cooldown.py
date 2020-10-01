# Generates directories, input files and batch scripts for a cooldown, i.e. a series of dca/analysis
# runs with gradually decreased temperature.
# By default, this script and the provided template input files (input_sp.json.in and
# input_sp.json.in) are configured for DCA(+) calculations of the 2D single-band Hubbard model with
# on-site Coulomb interaction U and fixed density d.
#
# Usage: 1. Configure the EDIT block.
#        2. Execute the script: python cooldown.py
#
# See https://github.com/CompFUSE/DCA/wiki/Running for more details on how to use this script and
# how to run a DCA(+) calculation.
#
# Author: Urs R. Haehner (haehneru@itp.phys.ethz.ch)

import os
import sys
import argparse

class CoolDownArguments():
    def __init__(self):
        pass
    
    def getOptions(self,args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--job_template",
                           dest="batch_tmpl",
                           help="path to job template",
                           default=None)
        parser.add_argument("--run_command",
                           dest="run_command_dca",
                           help="for dca srun -your args",
                           default=None)
        parser.add_argument("--run_command_analysis",
                           dest="run_command_analysis",
                           help="for analysis srun -your args",
                           default=None)
        parser.add_argument("--main_dca",
                           dest="main_dca",
                           help="path to main_dca",
                           default="./main_dca")
        parser.parse_args(namespace=self);
        
        ################################################################################
        ##################################### EDIT #####################################
        ################################################################################

        # The file should contain the placeholder JOBS.
        # Other optional placeholders: APPLICATION, HUBBARDU, DENS, SIZE.
        
        # Simulation parameters
        self.dca_plus = False  # True|False
        self.U = 6             # on-site Coulomb interaction
        self.d = 0.95          # density

        # DCA cluster (see cluster_definitions.txt)
        self.Nc = 4 # Cluster size
        self.cluster_vec_1 = [2, 0]
        self.cluster_vec_2 = [0, 2]

        # DCA(+) iterations
        self.iters_ht = 8  # first/highest temperature
        self.iters = 6     # other temperatures

        # Temperatures for cooldown
        self.temps = [1, 0.75, 0.5, 0.25, 0.125, 0.1, 0.09, 0.08, 0.07]
        # Starting temperature for computing two-particle quantities and running the analysis.
        self.T_analysis = 0.1

# Formats the inverse temperature beta.
def beta_format(x):
    return ('%.6g' % x)

# Prepares the input file for the given temperature.
def prepare_input_file(cda, filename, T_ind):
    file = open(filename, 'r')
    text = file.read()
    file.close()

    text = text.replace('CURRENT_TEMP', str(cda.temps[T_ind]))
    text = text.replace('BETA', str(beta_format(1./cda.temps[T_ind])))
    text = text.replace('DENS', str(cda.d))
    text = text.replace('HUBBARDU', str(cda.U))

    # For the first temperature set the initial self-energy to zero.
    if (T_ind == 0):
        text = text.replace('./T=PREVIOUS_TEMP/dca_sp.hdf5', 'zero')
    else:
        text = text.replace('PREVIOUS_TEMP', str(cda.temps[T_ind-1]))

    if (T_ind == 0):
        text = text.replace('ITERS', str(cda.iters_ht))
    else:
        text = text.replace('ITERS', str(cda.iters))

    # Use lower() since Python's booleans start with a capital letter, while JSON's booleans don't.
    text = text.replace('DO_DCA_PLUS', str(cda.dca_plus).lower())

    text = text.replace('VEC1', str(cda.cluster_vec_1))
    text = text.replace('VEC2', str(cda.cluster_vec_2))

    file = open(filename, 'w')
    file.write(text)
    file.close()

################################################################################

if __name__ == '__main__':
    batch_str_dca = ''
    batch_str_analysis = ''

    cda = CoolDownArguments()
    cda.getOptions(sys.argv[1:])
    
    print('Generating directories and input files:')

    for T_ind, T in enumerate(cda.temps):
        print('T = ' + str(T))

        # Create directory.
        dir_str = './T=' + str(T)
        cmd = 'mkdir -p ' + dir_str
        os.system(cmd)
        cmd = 'mkdir -p ' + dir_str + "/configuration"
        os.system(cmd)

        input_sp = dir_str + '/input_sp.json'
        input_tp = dir_str + '/input_tp.json'

        data_dca_sp   = dir_str + '/dca_sp.hdf5'
        data_dca_tp   = dir_str + '/dca_tp.hdf5'
        data_analysis = dir_str + '/analysis.hdf5'

        # dca sp
        # Generate the sp input file.
        cmd = 'cp ./input_sp.json.in ' + input_sp
        os.system(cmd)
        prepare_input_file(cda, input_sp, T_ind)

        # Add job.
        batch_str_dca = "{} {} {} {} \n".format(batch_str_dca, cda.run_command_dca, cda.main_dca, input_sp)

        if (T <= cda.T_analysis):
            # dca tp
            # Generate the tp input file.
            cmd = 'cp ./input_tp.json.in ' + input_tp
            os.system(cmd)
            prepare_input_file(cda, input_tp, T_ind)

            # Add job.
            batch_str_dca = "{} {} {} {}\n".format(batch_str_dca, cda.run_command_dca, cda.main_dca, input_tp)

            # analysis
            # Add job.
            batch_str_analysis = "{} {} {} {}\n".format(batch_str_analysis, cda.run_command_analysis, cda.main_dca, input_tp)


    # Get filename extension of batch script.
    _, extension = os.path.splitext(cda.batch_tmpl)

    # Generate the dca batch script.
    batch_name_dca = 'job.dca_U=' + str(cda.U) + '_d=' + str(cda.d) + '_Nc=' + str(cda.Nc) + extension
    print('\nGenerating the dca batch script: ' + batch_name_dca)
    file = open(cda.batch_tmpl, 'r')
    text_dca = file.read()
    file.close()

    text_dca = text_dca.replace('APPLICATION', 'dca')
    text_dca = text_dca.replace('HUBBARDU', str(cda.U))
    text_dca = text_dca.replace('DENS', str(cda.d))
    text_dca = text_dca.replace('SIZE', str(cda.Nc))
    text_dca = text_dca.replace('JOBS', batch_str_dca)

    file = open(batch_name_dca, 'w')
    file.write(text_dca)
    file.close()

    # Generate the analysis batch script.
    batch_name_analysis = 'job.analysis_U=' + str(cda.U) + '_d=' + str(cda.d) + '_Nc=' + str(cda.Nc) + extension
    print('Generating the analysis batch script: ' + batch_name_analysis)
    file = open(cda.batch_tmpl, 'r')
    text_analysis = file.read()
    file.close()

    text_analysis = text_analysis.replace('APPLICATION', 'analysis')
    text_analysis = text_analysis.replace('HUBBARDU', str(cda.U))
    text_analysis = text_analysis.replace('DENS', str(cda.d))
    text_analysis = text_analysis.replace('SIZE', str(cda.Nc))
    text_analysis = text_analysis.replace('JOBS', batch_str_analysis)

    file = open(batch_name_analysis, 'w')
    file.write(text_analysis)
    file.close()

    # Make batch scripts executable.
    os.chmod(batch_name_dca, 0o755)
    os.chmod(batch_name_analysis, 0o755)
