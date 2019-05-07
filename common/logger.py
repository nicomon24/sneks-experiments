'''
    Logging utils. The logger integrates stdout logging, tensorboard logging and
    sacred logging.
'''
from tensorboardX import SummaryWriter

class Logger():

    def __init__(self, experiment_name, loggers=[], **kwargs):
        # Set base directory
        self.BASE_DIR = 'runs/'
        # Save experiment name
        self.experiment_name = experiment_name
        # Create each of the loggers
        self.loggers = {}
        for t in loggers:
            if t == 'tensorboard':
                self.loggers['tensorboard'] = SummaryWriter('runs/' + experiment_name)
            elif t == 'sacred':
                assert kwargs['sacred_run'] is not None, "Need to pass a sacred run."
                self.loggers['sacred'] = kwargs['sacred_run']
            else:
                raise Exception('Unrecognized logger.')

    def log_kv(self, key, value, index):
        for logger_type, logger in self.loggers.items():
            if logger_type == 'tensorboard':
                logger.add_scalar(key, value, index)
            elif logger_type == 'sacred':
                logger.log_scalar(key, value)
            else:
                raise Exception('Unrecognized logger.')
